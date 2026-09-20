import copy
from dataclasses import dataclass
from typing import Union, Optional, List, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from hnet.modules.isotropic import Isotropic, IsotropicInferenceParams
from hnet.modules.dc import (
    RoutingModule,
    ChunkLayer,
    DeChunkLayer,
    RoutingModuleState,
    DeChunkState,
)
from hnet.modules.utils import apply_optimization_params

from .config_hnet import HNetConfig


class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.ones_like(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = grad_output
        return grad_x

def ste_func(x):
    return STE.apply(x)


@dataclass
class HNetState:
    encoder_state: Optional[IsotropicInferenceParams] = None
    routing_module_state: Optional[Union[RoutingModuleState, List[RoutingModuleState]]] = None
    main_network_state: Optional[
        Union["HNetState", IsotropicInferenceParams, List[Union["HNetState", IsotropicInferenceParams]]]
    ] = None
    dechunk_state: Optional[Union[DeChunkState, List[DeChunkState]]] = None
    decoder_state: Optional[IsotropicInferenceParams] = None


def is_stage_layout(layout):
    """
    Checks if a layout specification represents a single stage:
    either [encoder, mid, decoder] or [innermost_isotropic] or a string.
    """
    if isinstance(layout, str):
        return True
    if isinstance(layout, list):
        if len(layout) == 1 and isinstance(layout[0], str):
            return True
        if len(layout) == 3 and isinstance(layout[0], str) and isinstance(layout[2], str):
            return True
    return False


def parse_backbone_layouts(mid, num_backbones=None):
    """
    Parses the middle layout specification into a list of individual backbone layouts.
    Supports both symmetric specifications (single layout replicated N times)
    and asymmetric specifications (list of distinct backbone layouts).
    """
    if is_stage_layout(mid):
        single_layout = [mid] if isinstance(mid, str) else mid
        if num_backbones is not None and isinstance(num_backbones, int) and num_backbones > 1:
            return [copy.deepcopy(single_layout) for _ in range(num_backbones)]
        return [single_layout]

    if isinstance(mid, list):
        layouts = []
        for item in mid:
            if isinstance(item, str):
                layouts.append([item])
            else:
                layouts.append(item)
        return layouts

    raise ValueError(f"Unable to parse backbone layouts from: {mid}")


class HNet(nn.Module):
    def __init__(
        self,
        config: HNetConfig,
        stage_idx: int,
        layout: Optional[Union[str, List]] = None,
        routing_layer: Optional[Union[nn.Module, List[nn.Module]]] = None,
        num_backbones: Optional[Union[int, List[int]]] = None,
        join_mode: Optional[Union[str, List[str]]] = None,
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.stage_idx = stage_idx
        d_model_idx = min(stage_idx, len(config.d_model) - 1)
        self.d_model = config.d_model[d_model_idx]

        if routing_layer is None:
            routing_layer = (
                kwargs.get("routing_transform")
                or kwargs.get("routing_head")
                or kwargs.get("router_layer")
            )
        if num_backbones is None:
            num_backbones = kwargs.get(
                "n_backbones", getattr(config, "num_backbones", None)
            )
        if join_mode is None:
            join_mode = kwargs.get(
                "backbone_join_mode", getattr(config, "join_mode", "independent")
            )

        if isinstance(join_mode, list):
            curr_join_mode = (
                join_mode[stage_idx] if stage_idx < len(join_mode) else "independent"
            )
        else:
            curr_join_mode = str(join_mode) if join_mode is not None else "independent"

        curr_join_mode = curr_join_mode.lower()
        if curr_join_mode in ("independent", "concat", "independent_ema", "mode1", "1"):
            self.join_mode = "independent"
        elif curr_join_mode in ("shared", "single", "shared_ema", "single_stream", "same_stream", "mode2", "2"):
            self.join_mode = "single_stream"
        else:
            raise ValueError(f"Unknown join_mode: {curr_join_mode}")

        if layout is None:
            arch_layout = config.arch_layout
            for _ in range(stage_idx):
                arch_layout = arch_layout[1]
            layout = arch_layout

        if isinstance(layout, str):
            layout = [layout]

        while isinstance(layout, list) and len(layout) == 1 and isinstance(layout[0], list):
            layout = layout[0]

        assert isinstance(layout, list), f"Wrong arch_layout: {layout}"
        if len(layout) == 3 and isinstance(layout[0], str) and isinstance(layout[2], str):
            self.is_innermost = False
            encoder_layout = layout[0]
            main_layout = layout[1]
            decoder_layout = layout[2]
        elif len(layout) == 1 and isinstance(layout[0], str):
            self.is_innermost = True
            main_layout = layout[0]
            encoder_layout = None
            decoder_layout = None
        else:
            raise NotImplementedError(f"Unrecognized layout at stage {stage_idx}: {layout}")

        if self.is_innermost:
            self.n_backbones = 1
            self.main_networks = nn.ModuleList([
                Isotropic(
                    config=config,
                    stage_idx=stage_idx,
                    pos_idx=0,
                    layout=main_layout,
                    **factory_kwargs,
                )
            ])
            self.main_network = self.main_networks[0]
        else:
            self.encoder = Isotropic(
                config=config,
                stage_idx=stage_idx,
                pos_idx=0,
                layout=encoder_layout,
                **factory_kwargs,
            )
            self.decoder = Isotropic(
                config=config,
                stage_idx=stage_idx,
                pos_idx=2,
                layout=decoder_layout,
                **factory_kwargs,
            )

            # Determine backbones and their layouts
            stage_num_bb = (
                num_backbones[stage_idx]
                if isinstance(num_backbones, list) and stage_idx < len(num_backbones)
                else (num_backbones if isinstance(num_backbones, int) else None)
            )
            backbone_layouts = parse_backbone_layouts(main_layout, num_backbones=stage_num_bb)
            self.n_backbones = len(backbone_layouts)

            # Build backbones for this depth
            main_networks = []
            for i in range(self.n_backbones):
                sub_model = HNet(
                    config=config,
                    stage_idx=stage_idx + 1,
                    layout=backbone_layouts[i],
                    routing_layer=routing_layer,
                    num_backbones=num_backbones,
                    join_mode=join_mode,
                    **factory_kwargs,
                )
                main_networks.append(sub_model)
            self.main_networks = nn.ModuleList(main_networks)
            self.main_network = self.main_networks[0]

            # Build token-wise routing layers
            if routing_layer is None:
                routing_layers = [nn.Identity() for _ in range(self.n_backbones)]
            elif isinstance(routing_layer, (list, nn.ModuleList)):
                routing_layers = [
                    routing_layer[i] if i < len(routing_layer) else nn.Identity()
                    for i in range(self.n_backbones)
                ]
            elif isinstance(routing_layer, nn.Module):
                if isinstance(routing_layer, nn.Identity):
                    routing_layers = [nn.Identity() for _ in range(self.n_backbones)]
                else:
                    routing_layers = [
                        copy.deepcopy(routing_layer) for _ in range(self.n_backbones)
                    ]
            elif callable(routing_layer):
                try:
                    routing_layers = [routing_layer(self.d_model) for _ in range(self.n_backbones)]
                except TypeError:
                    routing_layers = [routing_layer() for _ in range(self.n_backbones)]
            else:
                routing_layers = [nn.Identity() for _ in range(self.n_backbones)]
            self.routing_layers = nn.ModuleList(routing_layers)
            self.routing_layer = self.routing_layers[0]

            stage_N = (
                config.N[stage_idx]
                if isinstance(getattr(config, "N", None), (list, tuple)) and stage_idx < len(config.N)
                else getattr(config, "N", 2.0)
            )
            # Routing modules & Chunk layers per backbone
            self.routing_modules = nn.ModuleList([
                RoutingModule(
                    self.d_model,
                    stage_idx=stage_idx,
                    backbone_idx=i,
                    n_backbones=self.n_backbones,
                    target_N=stage_N,
                    **factory_kwargs,
                )
                for i in range(self.n_backbones)
            ])
            self.routing_module = self.routing_modules[0]

            self.chunk_layers = nn.ModuleList([
                ChunkLayer() for _ in range(self.n_backbones)
            ])
            self.chunk_layer = self.chunk_layers[0]

            # DeChunk layers & projection
            if self.join_mode == "independent":
                self.dechunk_layers = nn.ModuleList([
                    DeChunkLayer(self.d_model) for _ in range(self.n_backbones)
                ])
                self.dechunk_layer = self.dechunk_layers[0]
                if self.n_backbones > 1:
                    self.backbone_proj = nn.Linear(
                        self.n_backbones * self.d_model, self.d_model, **factory_kwargs
                    )
                else:
                    self.backbone_proj = nn.Identity()
            else:
                self.dechunk_layer = DeChunkLayer(self.d_model)
                self.dechunk_layers = nn.ModuleList([self.dechunk_layer])
                self.backbone_proj = nn.Identity()

            # Residual projection in fp32
            self.residual_proj = nn.Linear(
                self.d_model, self.d_model, device=device, dtype=torch.float32
            )
            nn.init.zeros_(self.residual_proj.weight)
            self.residual_proj.weight._no_reinit = True

            self.residual_func = lambda out, residual, p: out * ste_func(p) + residual

        prev_d_model = (
            config.d_model[min(stage_idx - 1, len(config.d_model) - 1)]
            if stage_idx > 0
            else self.d_model
        )
        if stage_idx > 0 and self.d_model - prev_d_model > 0:
            self.pad_dimension = nn.Parameter(
                torch.zeros(
                    self.d_model - prev_d_model, **factory_kwargs
                )
            )
        else:
            self.pad_dimension = None

    def _init_weights(self, initializer_range: float = 0.02, parent_residuals: int = 0) -> None:
        n_residuals = parent_residuals
        if self.is_innermost:
            n_residuals += self.main_networks[0].height
            for main_network in self.main_networks:
                for name, m in main_network.named_modules():
                    if isinstance(m, nn.Linear) and not getattr(m.weight, "_no_reinit", False):
                        if "out_proj" in name or "fc2" in name:
                            nn.init.normal_(m.weight, mean=0.0, std=initializer_range / (n_residuals ** 0.5))
                        else:
                            nn.init.normal_(m.weight, mean=0.0, std=initializer_range)
        else:
            n_residuals += self.encoder.height + self.decoder.height
            for name, m in self.encoder.named_modules():
                if isinstance(m, nn.Linear) and not getattr(m.weight, "_no_reinit", False):
                    if "out_proj" in name or "fc2" in name:
                        nn.init.normal_(m.weight, mean=0.0, std=initializer_range / (n_residuals ** 0.5))
                    else:
                        nn.init.normal_(m.weight, mean=0.0, std=initializer_range)
            for name, m in self.decoder.named_modules():
                if isinstance(m, nn.Linear) and not getattr(m.weight, "_no_reinit", False):
                    if "out_proj" in name or "fc2" in name:
                        nn.init.normal_(m.weight, mean=0.0, std=initializer_range / (n_residuals ** 0.5))
                    else:
                        nn.init.normal_(m.weight, mean=0.0, std=initializer_range)

            if isinstance(self.backbone_proj, nn.Linear):
                nn.init.normal_(self.backbone_proj.weight, mean=0.0, std=initializer_range)
                if self.backbone_proj.bias is not None:
                    nn.init.zeros_(self.backbone_proj.bias)

            for routing_layer in self.routing_layers:
                for name, m in routing_layer.named_modules():
                    if isinstance(m, nn.Linear) and not getattr(m.weight, "_no_reinit", False):
                        nn.init.normal_(m.weight, mean=0.0, std=initializer_range)

            for main_network in self.main_networks:
                main_network._init_weights(initializer_range, n_residuals)

    def _apply_lr_multiplier(self, lr_multiplier: list[float]) -> None:
        """
        Applies the learning rate multipliers to the parameters of the model.
        """
        mult_idx = min(self.stage_idx, len(lr_multiplier) - 1) if len(lr_multiplier) > 0 else 0
        mult = lr_multiplier[mult_idx] if len(lr_multiplier) > 0 else 1.0
        for param in self.parameters():
            apply_optimization_params(param, lr_multiplier=mult)

        if not self.is_innermost:
            for main_network in self.main_networks:
                main_network._apply_lr_multiplier(lr_multiplier)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        if self.is_innermost:
            return HNetState(
                main_network_state=self.main_networks[0].allocate_inference_cache(
                    batch_size, max_seqlen, dtype=dtype
                )
            )
        else:
            device = self.residual_proj.weight.device
            routing_module_states = [
                rm.allocate_inference_cache(batch_size, max_seqlen, device, dtype=dtype)
                for rm in self.routing_modules
            ]
            main_network_states = [
                mn.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype)
                for mn in self.main_networks
            ]
            if self.join_mode == "independent":
                dechunk_states = [
                    dl.allocate_inference_cache(batch_size, max_seqlen, device, dtype=dtype)
                    for dl in self.dechunk_layers
                ]
            else:
                dechunk_states = self.dechunk_layer.allocate_inference_cache(
                    batch_size, max_seqlen, device, dtype=dtype
                )

            return HNetState(
                encoder_state=self.encoder.allocate_inference_cache(
                    batch_size, max_seqlen, dtype=dtype
                ),
                routing_module_state=routing_module_states,
                main_network_state=main_network_states,
                dechunk_state=dechunk_states,
                decoder_state=self.decoder.allocate_inference_cache(
                    batch_size, max_seqlen, dtype=dtype
                ),
            )

    def forward(
        self,
        hidden_states,
        cu_seqlens=None,
        max_seqlen=None,
        mask=None,
        inference_params=None,
        **mixer_kwargs,
    ):
        assert mask is not None or (
            cu_seqlens is not None and max_seqlen is not None
        ), "Either mask or cu_seqlens and max_seqlen must be provided"

        if inference_params is None:
            inference_params = HNetState(main_network_state=None)
        else:
            assert (
                mask is not None
            ), "Mask must be provided if inference_params is provided"

        D = hidden_states.shape[-1]
        EARLY_DIMS = hidden_states.shape[:-1]

        if self.pad_dimension is not None:
            hidden_states = torch.cat(
                (hidden_states, self.pad_dimension.expand(EARLY_DIMS + (-1,))), dim=-1
            )

        if self.is_innermost:
            mn_state = (
                inference_params.main_network_state[0]
                if isinstance(inference_params.main_network_state, list)
                else inference_params.main_network_state
            )
            hidden_states = self.main_networks[0](
                hidden_states,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                mask=mask,
                inference_params=mn_state,
                **mixer_kwargs,
            )
            hidden_states = hidden_states[..., :D]
            return hidden_states, []

        encoder_out = self.encoder(
            hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            mask=mask,
            inference_params=inference_params.encoder_state,
            **mixer_kwargs,
        )

        hidden_states_for_residual = encoder_out.to(
            dtype=self.residual_proj.weight.dtype
        )
        residual = self.residual_proj(hidden_states_for_residual)

        bpred_outputs = []
        backbone_outputs = []
        all_prev_bpreds = []
        next_cu_seqlens_list = []

        for i in range(self.n_backbones):
            rm_state = (
                inference_params.routing_module_state[i]
                if isinstance(inference_params.routing_module_state, list)
                else inference_params.routing_module_state
            )
            # Token-wise layer between final encoder transformer embeddings and routing score calculation
            routing_input_i = self.routing_layers[i](encoder_out)
            bpred_output_i = self.routing_modules[i](
                routing_input_i,
                cu_seqlens=cu_seqlens,
                mask=mask,
                inference_params=rm_state,
            )
            bpred_outputs.append(bpred_output_i)

            # Aggregation for backbone is taken from final encoder embeddings before the new head
            chunk_hidden_states_i, next_cu_seqlens_i, next_max_seqlen_i, next_mask_i = self.chunk_layers[i](
                encoder_out, bpred_output_i.boundary_mask, cu_seqlens, mask=mask
            )
            next_cu_seqlens_list.append(next_cu_seqlens_i)

            mn_state = (
                inference_params.main_network_state[i]
                if isinstance(inference_params.main_network_state, list)
                else inference_params.main_network_state
            )
            if chunk_hidden_states_i.numel() > 0 and (
                (cu_seqlens is not None and chunk_hidden_states_i.shape[0] > 0)
                or (cu_seqlens is None and chunk_hidden_states_i.shape[1] > 0)
            ):
                backbone_out_i, prev_bpred_i = self.main_networks[i](
                    chunk_hidden_states_i,
                    cu_seqlens=next_cu_seqlens_i,
                    max_seqlen=next_max_seqlen_i,
                    mask=next_mask_i,
                    inference_params=mn_state,
                    **mixer_kwargs,
                )
            else:
                backbone_out_i = chunk_hidden_states_i
                prev_bpred_i = []

            backbone_outputs.append(backbone_out_i)
            all_prev_bpreds.extend(prev_bpred_i)

        if self.join_mode == "independent":
            # Mode 1: Independent EMA stream per backbone
            dechunked_outs = []
            for i in range(self.n_backbones):
                dc_state = (
                    inference_params.dechunk_state[i]
                    if isinstance(inference_params.dechunk_state, list)
                    else inference_params.dechunk_state
                )
                dechunked_i = self.dechunk_layers[i](
                    backbone_outputs[i],
                    bpred_outputs[i].boundary_mask,
                    bpred_outputs[i].boundary_prob,
                    next_cu_seqlens_list[i],
                    mask=mask,
                    inference_params=dc_state,
                )
                dechunked_outs.append(dechunked_i)

            if self.n_backbones == 1:
                joined = dechunked_outs[0]
                selected_probs = bpred_outputs[0].selected_probs
            else:
                concat_out = torch.cat(dechunked_outs, dim=-1)
                joined = self.backbone_proj(concat_out)
                selected_probs = torch.stack(
                    [bp.selected_probs for bp in bpred_outputs], dim=0
                ).mean(dim=0)

            hidden_states = self.residual_func(
                joined.to(dtype=residual.dtype), residual, selected_probs
            ).to(joined.dtype)

        else:
            # Mode 2: Single shared EMA stream
            dc_state = (
                inference_params.dechunk_state[0]
                if isinstance(inference_params.dechunk_state, list)
                else inference_params.dechunk_state
            )
            if self.n_backbones == 1:
                joined = self.dechunk_layer(
                    backbone_outputs[0],
                    bpred_outputs[0].boundary_mask,
                    bpred_outputs[0].boundary_prob,
                    next_cu_seqlens_list[0],
                    mask=mask,
                    inference_params=dc_state,
                )
                selected_probs = bpred_outputs[0].selected_probs
            else:
                device = encoder_out.device
                dtype = encoder_out.dtype
                D_backbone = encoder_out.shape[-1]

                if cu_seqlens is not None:
                    # Packed mode: encoder_out is (T, D)
                    T = encoder_out.shape[0]
                    full_v_list = []
                    for i in range(self.n_backbones):
                        full_v_i = torch.zeros((T, D_backbone), device=device, dtype=dtype)
                        full_v_i[bpred_outputs[i].boundary_mask] = backbone_outputs[i]
                        full_v_list.append(full_v_i)

                    active_masks = torch.stack([bp.boundary_mask for bp in bpred_outputs], dim=0)  # (B, T)
                    scores_stack = torch.stack([bp.boundary_prob[:, 1] for bp in bpred_outputs], dim=0)  # (B, T)
                    active_scores = scores_stack * active_masks.to(scores_stack.dtype)  # (B, T)
                    b_count = active_masks.sum(dim=0).float()  # (T,)
                    sum_scores = active_scores.sum(dim=0)  # (T,)
                    score = sum_scores / b_count.clamp(min=1.0)  # (T,)

                    full_v_stack = torch.stack(full_v_list, dim=0)  # (B, T, D)
                    weighted_v = (active_scores.unsqueeze(-1) * full_v_stack).sum(dim=0)  # (T, D)
                    V = weighted_v / sum_scores.clamp(min=1e-8).unsqueeze(-1)  # (T, D)

                    union_mask = active_masks.any(dim=0)  # (T,) bool
                    union_cu_seqlens = F.pad(union_mask.cumsum(dim=0)[cu_seqlens[1:] - 1], (1, 0))
                    union_V = V[union_mask]
                    union_boundary_prob = torch.stack([1.0 - score, score], dim=-1)

                    joined = self.dechunk_layer(
                        union_V,
                        union_mask,
                        union_boundary_prob,
                        union_cu_seqlens,
                        mask=None,
                        inference_params=dc_state,
                    )
                    selected_probs = score.unsqueeze(-1)

                else:
                    # Unpacked mode: encoder_out is (B_batch, L, D)
                    B_batch, L, _ = encoder_out.shape
                    full_v_list = []
                    for i in range(self.n_backbones):
                        mask_i = bpred_outputs[i].boundary_mask
                        token_idx_i = torch.arange(L, device=device)[None, :] + (~mask_i).long() * L
                        seq_sorted_indices_i = torch.argsort(token_idx_i, dim=1)
                        full_v_i = torch.zeros((B_batch, L, D_backbone), device=device, dtype=dtype)
                        full_v_i.scatter_(
                            dim=1,
                            index=seq_sorted_indices_i[:, :backbone_outputs[i].shape[1], None].expand(-1, -1, D_backbone),
                            src=backbone_outputs[i],
                        )
                        full_v_list.append(full_v_i)

                    active_masks = torch.stack([bp.boundary_mask for bp in bpred_outputs], dim=0)  # (B_backbones, B_batch, L)
                    scores_stack = torch.stack([bp.boundary_prob[..., 1] for bp in bpred_outputs], dim=0)  # (B_backbones, B_batch, L)
                    active_scores = scores_stack * active_masks.to(scores_stack.dtype)  # (B_backbones, B_batch, L)
                    b_count = active_masks.sum(dim=0).float()  # (B_batch, L)
                    sum_scores = active_scores.sum(dim=0)  # (B_batch, L)
                    score = sum_scores / b_count.clamp(min=1.0)  # (B_batch, L)

                    full_v_stack = torch.stack(full_v_list, dim=0)  # (B_backbones, B_batch, L, D)
                    weighted_v = (active_scores.unsqueeze(-1) * full_v_stack).sum(dim=0)  # (B_batch, L, D)
                    V = weighted_v / sum_scores.clamp(min=1e-8).unsqueeze(-1)  # (B_batch, L, D)

                    union_mask = active_masks.any(dim=0)  # (B_batch, L) bool
                    if mask is not None:
                        union_mask = union_mask & mask
                    num_tokens = union_mask.sum(dim=-1)
                    union_max_seqlen = int(num_tokens.max())
                    token_idx = torch.arange(L, device=device)[None, :] + (~union_mask).long() * L
                    seq_sorted_indices = torch.argsort(token_idx, dim=1)
                    union_V = torch.gather(
                        V,
                        dim=1,
                        index=seq_sorted_indices[:, :union_max_seqlen, None].expand(-1, -1, D_backbone),
                    )
                    union_boundary_prob = torch.stack([1.0 - score, score], dim=-1)

                    joined = self.dechunk_layer(
                        union_V,
                        union_mask,
                        union_boundary_prob,
                        cu_seqlens=None,
                        mask=mask,
                        inference_params=dc_state,
                    )
                    selected_probs = score.unsqueeze(-1)

            hidden_states = self.residual_func(
                joined.to(dtype=residual.dtype), residual, selected_probs
            ).to(joined.dtype)

        hidden_states = self.decoder(
            hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            mask=mask,
            inference_params=inference_params.decoder_state,
            **mixer_kwargs,
        )

        hidden_states = hidden_states[..., :D]
        return hidden_states, [*bpred_outputs, *all_prev_bpreds]

    def step(self, hidden_states, inference_params):
        D = hidden_states.shape[-1]

        if self.pad_dimension is not None:
            hidden_states = torch.cat(
                (
                    hidden_states,
                    self.pad_dimension.expand(hidden_states.shape[:-1] + (-1,)),
                ),
                dim=-1,
            )

        if self.is_innermost:
            mn_state = (
                inference_params.main_network_state[0]
                if isinstance(inference_params.main_network_state, list)
                else inference_params.main_network_state
            )
            hidden_states = self.main_networks[0].step(hidden_states, mn_state)
            hidden_states = hidden_states[..., :D]
            return hidden_states, []

        encoder_out = self.encoder.step(hidden_states, inference_params.encoder_state)
        hidden_states_for_residual = encoder_out.to(
            dtype=self.residual_proj.weight.dtype
        )
        residual = self.residual_proj(hidden_states_for_residual)

        bpred_outputs = []
        backbone_outputs = []
        all_prev_bpreds = []

        for i in range(self.n_backbones):
            rm_state = (
                inference_params.routing_module_state[i]
                if isinstance(inference_params.routing_module_state, list)
                else inference_params.routing_module_state
            )
            # Token-wise layer between final encoder transformer embeddings and routing score calculation
            routing_input_i = self.routing_layers[i](encoder_out)
            bpred_output_i = self.routing_modules[i].step(routing_input_i, rm_state)
            bpred_outputs.append(bpred_output_i)

            # Aggregation for backbone is taken from final encoder embeddings before the new head
            hidden_states_inner_i = self.chunk_layers[i].step(
                encoder_out, bpred_output_i.boundary_mask
            )

            mn_state = (
                inference_params.main_network_state[i]
                if isinstance(inference_params.main_network_state, list)
                else inference_params.main_network_state
            )
            if hidden_states_inner_i.shape[0] > 0:
                backbone_out_i, prev_bpred_i = self.main_networks[i].step(
                    hidden_states_inner_i, mn_state
                )
            else:
                backbone_out_i = None
                prev_bpred_i = []

            backbone_outputs.append(backbone_out_i)
            all_prev_bpreds.extend(prev_bpred_i)

        if self.join_mode == "independent":
            # Mode 1: Independent EMA stream per backbone
            dechunked_outs = []
            for i in range(self.n_backbones):
                dc_state = (
                    inference_params.dechunk_state[i]
                    if isinstance(inference_params.dechunk_state, list)
                    else inference_params.dechunk_state
                )
                inner_input = (
                    backbone_outputs[i]
                    if backbone_outputs[i] is not None
                    else torch.zeros(
                        0, 1, self.d_model, device=hidden_states.device, dtype=hidden_states.dtype
                    )
                )
                dechunked_i = self.dechunk_layers[i].step(
                    inner_input,
                    bpred_outputs[i].boundary_mask,
                    bpred_outputs[i].boundary_prob,
                    dc_state,
                )
                dechunked_outs.append(dechunked_i)

            if self.n_backbones == 1:
                joined = dechunked_outs[0]
                selected_probs = bpred_outputs[0].selected_probs
            else:
                concat_out = torch.cat(dechunked_outs, dim=-1)
                joined = self.backbone_proj(concat_out)
                selected_probs = torch.stack(
                    [bp.selected_probs for bp in bpred_outputs], dim=0
                ).mean(dim=0)

            hidden_states = self.residual_func(
                joined.to(dtype=residual.dtype), residual, selected_probs
            ).to(joined.dtype)

        else:
            # Mode 2: Single shared EMA stream
            dc_state = (
                inference_params.dechunk_state[0]
                if isinstance(inference_params.dechunk_state, list)
                else inference_params.dechunk_state
            )
            if self.n_backbones == 1:
                inner_input = (
                    backbone_outputs[0]
                    if backbone_outputs[0] is not None
                    else torch.zeros(
                        0, 1, self.d_model, device=hidden_states.device, dtype=hidden_states.dtype
                    )
                )
                joined = self.dechunk_layer.step(
                    inner_input,
                    bpred_outputs[0].boundary_mask,
                    bpred_outputs[0].boundary_prob,
                    dc_state,
                )
                selected_probs = bpred_outputs[0].selected_probs
            else:
                B_batch = hidden_states.shape[0]
                full_v_list = []
                active_scores_list = []
                for i in range(self.n_backbones):
                    full_v_i = torch.zeros(
                        (B_batch, self.d_model),
                        device=hidden_states.device,
                        dtype=hidden_states.dtype,
                    )
                    mask_i = bpred_outputs[i].boundary_mask
                    if backbone_outputs[i] is not None and mask_i.any():
                        full_v_i[mask_i] = backbone_outputs[i].squeeze(1)
                    full_v_list.append(full_v_i)

                    score_i = bpred_outputs[i].boundary_prob[:, 1].clamp(
                        min=1e-4, max=1.0 - 1e-4
                    )
                    active_scores_i = score_i * mask_i.to(dtype=score_i.dtype)
                    active_scores_list.append(active_scores_i)

                active_masks = torch.stack(
                    [bp.boundary_mask for bp in bpred_outputs], dim=0
                )  # (N_bb, B_batch)
                any_active = active_masks.any(dim=0)  # (B_batch,) bool
                b_count = active_masks.sum(dim=0).float()  # (B_batch,)

                active_scores_stack = torch.stack(
                    active_scores_list, dim=0
                )  # (N_bb, B_batch)
                sum_scores = active_scores_stack.sum(dim=0)  # (B_batch,)
                score_val = sum_scores / b_count.clamp(min=1.0)  # (B_batch,)

                full_v_stack = torch.stack(full_v_list, dim=0)  # (N_bb, B_batch, D)
                weighted_v = (active_scores_stack.unsqueeze(-1) * full_v_stack).sum(
                    dim=0
                )  # (B_batch, D)
                V = weighted_v / sum_scores.clamp(min=1e-8).unsqueeze(-1)

                u = dc_state.last_value
                s = score_val.unsqueeze(-1)
                updated_val = (1.0 - s) * u + s * V
                result_val = torch.where(any_active.unsqueeze(-1), updated_val, u)
                dc_state.last_value.copy_(result_val)
                result = result_val.unsqueeze(1)

                selected_probs = torch.where(
                    any_active, score_val, torch.zeros_like(score_val)
                ).unsqueeze(-1)
                joined = result

            hidden_states = self.residual_func(
                joined.to(dtype=residual.dtype), residual, selected_probs
            ).to(joined.dtype)

        hidden_states = self.decoder.step(hidden_states, inference_params.decoder_state)
        hidden_states = hidden_states[..., :D]

        return hidden_states, [*bpred_outputs, *all_prev_bpreds]
