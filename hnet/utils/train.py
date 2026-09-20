"""
This file contains utility functions for training.

NOTE: This file is not used inside the HNet package, but contains useful utilities for training the model itself.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Union
import torch
import torch.nn as nn
import torch.distributed as dist

from hnet.modules.dc import RoutingModule, RoutingModuleOutput
from hnet.modules.utils import apply_optimization_params


def _compute_single_load_balancing_loss(
    router_output: RoutingModuleOutput,
    N_eff: float,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Computes 2-class load balancing loss for a single router output with target effective downsampling N_eff > 1.
    Formula:
        p* = 1 / N_eff
        L = [ (1 - true_ratio) * (1 - avg_prob) + true_ratio * avg_prob * (N_eff - 1) ] * N_eff / (N_eff - 1)

    Args:
        router_output: RoutingModuleOutput containing boundary_prob and boundary_mask.
        N_eff: Target effective downsampling factor (N * B). Must be > 1.0.
        mask: Optional boolean tensor indicating valid tokens (e.g. for padded batches).

    Returns:
        Scalar torch.Tensor representing the load balancing loss.
    """
    if N_eff <= 1.0:
        raise ValueError(f"Effective N must be > 1.0, got {N_eff}")

    boundary_prob = router_output.boundary_prob
    tokenized_prob = boundary_prob[..., -1]
    boundary_mask = router_output.boundary_mask

    # Resolve effective mask: use provided mask if shapes match, or router_output.mask if available
    effective_mask = None
    if mask is not None and mask.shape == boundary_mask.shape:
        effective_mask = mask.bool()
    elif getattr(router_output, "mask", None) is not None and router_output.mask.shape == boundary_mask.shape:
        effective_mask = router_output.mask.bool()

    if effective_mask is not None:
        if not effective_mask.any():
            return torch.tensor(0.0, device=boundary_prob.device, requires_grad=True)
        true_ratio = boundary_mask[effective_mask].float().mean()
        average_prob = tokenized_prob[effective_mask].float().mean()
    else:
        if boundary_mask.numel() == 0:
            return torch.tensor(0.0, device=boundary_prob.device, requires_grad=True)
        true_ratio = boundary_mask.float().mean()
        average_prob = tokenized_prob.float().mean()

    return (
        (1.0 - true_ratio) * (1.0 - average_prob)
        + true_ratio * average_prob * (N_eff - 1.0)
    ) * N_eff / (N_eff - 1.0)


def _group_router_outputs_by_stage(
    router_outputs: Union[
        RoutingModuleOutput,
        Sequence[Union[RoutingModuleOutput, Sequence[RoutingModuleOutput]]],
    ],
    B: Optional[Union[int, Sequence[int]]] = None,
) -> Dict[int, List[RoutingModuleOutput]]:
    """
    Groups router outputs by their hierarchical stage index.
    Handles:
      1. Nested sequences: [[stage0_r1, stage0_r2], [stage1_r1, ...]]
      2. Flattened sequences with router metadata (stage_idx on output or router_module)
      3. Flattened sequences with B specification: partitions using per-stage backbone counts
      4. Single RoutingModuleOutput
    """
    stage_groups: Dict[int, List[RoutingModuleOutput]] = defaultdict(list)

    if isinstance(router_outputs, RoutingModuleOutput):
        stage_idx = getattr(
            router_outputs,
            "stage_idx",
            getattr(router_outputs.router_module, "stage_idx", 0)
            if router_outputs.router_module is not None
            else 0,
        )
        stage_groups[stage_idx].append(router_outputs)
        return dict(stage_groups)

    if not isinstance(router_outputs, (list, tuple)):
        raise TypeError(f"Unsupported router_outputs type: {type(router_outputs)}")

    if len(router_outputs) == 0:
        return {}

    # Check if router_outputs is already a nested sequence of stages
    if all(isinstance(item, (list, tuple)) for item in router_outputs):
        for stage_idx, stage_list in enumerate(router_outputs):
            for r in stage_list:
                if isinstance(r, RoutingModuleOutput):
                    stage_groups[stage_idx].append(r)
        return dict(stage_groups)

    # Flat sequence: inspect each element for stage_idx metadata
    flat_list = [r for r in router_outputs if isinstance(r, RoutingModuleOutput)]
    has_stage_metadata = any(
        getattr(r, "stage_idx", None) is not None
        or (r.router_module is not None and getattr(r.router_module, "stage_idx", None) is not None)
        for r in flat_list
    )

    if has_stage_metadata:
        for r in flat_list:
            stage_idx = getattr(
                r,
                "stage_idx",
                getattr(r.router_module, "stage_idx", 0)
                if r.router_module is not None
                else 0,
            )
            stage_groups[stage_idx].append(r)
        return dict(stage_groups)

    # If B is provided as a sequence, partition flat_list by cumulative backbone counts
    if isinstance(B, (list, tuple)) and len(B) > 0:
        offset = 0
        curr_branches = 1
        for stage_idx, b_count in enumerate(B):
            num_routers_at_stage = curr_branches * b_count
            if offset >= len(flat_list):
                break
            stage_r = flat_list[offset : offset + num_routers_at_stage]
            stage_groups[stage_idx].extend(stage_r)
            offset += num_routers_at_stage
            curr_branches *= b_count
        if offset < len(flat_list):
            stage_groups[len(B)].extend(flat_list[offset:])
        return dict(stage_groups)

    # Default fallback: treat all as stage 0
    stage_groups[0].extend(flat_list)
    return dict(stage_groups)


def stage_load_balancing_loss(
    router_outputs: Union[RoutingModuleOutput, Sequence[RoutingModuleOutput]],
    N: float,
    B: Optional[int] = None,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Generalized load balancing loss for a stage with B backbones.
    Pushes each backbone of the stage towards activation ratio 1 / (N * B),
    where N is the desired sparsity ratio and B is the number of backbones at that stage.

    Args:
        router_outputs: A single RoutingModuleOutput or a sequence of RoutingModuleOutput
                        for the backbones belonging to this stage.
        N: Desired sparsity / compression ratio (e.g. 2.0, 4.0).
        B: Number of backbones at this stage. If None and a sequence is passed,
           inferred from router metadata or len(router_outputs).
        mask: Optional boolean tensor indicating valid tokens.

    Returns:
        A scalar torch.Tensor load balancing loss.
    """
    if isinstance(router_outputs, RoutingModuleOutput):
        b = B if B is not None else getattr(
            router_outputs,
            "n_backbones",
            getattr(router_outputs.router_module, "n_backbones", 1)
            if router_outputs.router_module is not None
            else 1,
        )
        n_eff = float(N * b)
        return _compute_single_load_balancing_loss(router_outputs, N_eff=n_eff, mask=mask)

    if isinstance(router_outputs, (list, tuple)):
        if len(router_outputs) == 0:
            raise ValueError("router_outputs cannot be empty")
        first_r = router_outputs[0]
        b = B if B is not None else getattr(
            first_r,
            "n_backbones",
            getattr(first_r.router_module, "n_backbones", len(router_outputs))
            if first_r.router_module is not None
            else len(router_outputs),
        )
        n_eff = float(N * b)
        losses = [
            _compute_single_load_balancing_loss(r, N_eff=n_eff, mask=mask)
            for r in router_outputs
            if isinstance(r, RoutingModuleOutput)
        ]
        if len(losses) == 0:
            return torch.tensor(0.0, requires_grad=True)
        return torch.stack(losses).mean()

    raise TypeError(f"Unsupported router_outputs type: {type(router_outputs)}")


def load_balancing_loss(
    router_output: Union[RoutingModuleOutput, Sequence[RoutingModuleOutput]],
    N: float,
    B: Optional[int] = None,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute the load balancing loss.
    Generalization of the original loss: handles both single RoutingModuleOutput
    and sequences of RoutingModuleOutput for multi-backbone stages.

    Args:
        router_output: The output(s) of the routing module(s).
        N: The sparsity factor (downsampling ratio). Must be > 1 / B.
        B: Optional number of backbones at the stage.
        mask: Optional boolean tensor indicating valid tokens.

    Returns:
        A single tensor representing the load balancing loss.
    """
    return stage_load_balancing_loss(router_outputs=router_output, N=N, B=B, mask=mask)


def hierarchical_load_balancing_loss(
    stage_router_outputs: Union[
        RoutingModuleOutput,
        Sequence[Union[RoutingModuleOutput, Sequence[RoutingModuleOutput]]],
    ],
    N: Union[float, Sequence[float]] = 2.0,
    B: Optional[Union[int, Sequence[int]]] = None,
    stage_weights: Optional[Sequence[float]] = None,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Computes load balancing loss across multiple hierarchy stages.
    Correctly disentangles stage router outputs when passed either as a structured
    nested list or as a flat list (e.g. out.bpred_output from HNetForCausalLM.forward).

    Args:
        stage_router_outputs: A nested list per stage, or a flattened sequence of RoutingModuleOutput.
        N: Sparsity ratio per stage (or single float shared across all stages).
        B: Number of backbones per stage (or single int / None).
        stage_weights: Optional weighting per stage for computing the total loss.
        mask: Optional boolean tensor indicating valid tokens.

    Returns:
        Mean (or weighted) load balancing loss across all stages.
    """
    stage_groups = _group_router_outputs_by_stage(stage_router_outputs, B=B)
    if not stage_groups:
        return torch.tensor(0.0, requires_grad=True)

    sorted_stages = sorted(stage_groups.keys())
    losses = []
    weights = []

    for stage_idx in sorted_stages:
        r_stage = stage_groups[stage_idx]
        n_val = (
            N[stage_idx]
            if isinstance(N, (list, tuple)) and stage_idx < len(N)
            else (N[-1] if isinstance(N, (list, tuple)) and len(N) > 0 else float(N))
        )
        b_val = (
            B[stage_idx]
            if isinstance(B, (list, tuple)) and stage_idx < len(B)
            else (B if isinstance(B, int) else None)
        )
        loss_i = stage_load_balancing_loss(r_stage, N=n_val, B=b_val, mask=mask)
        losses.append(loss_i)

        w = (
            stage_weights[stage_idx]
            if stage_weights is not None and stage_idx < len(stage_weights)
            else 1.0
        )
        weights.append(w)

    stacked_losses = torch.stack(losses)
    stacked_weights = torch.tensor(
        weights, device=stacked_losses.device, dtype=stacked_losses.dtype
    )
    return (stacked_losses * stacked_weights).sum() / stacked_weights.sum().clamp(min=1e-8)


def update_loss_free_bias(
    router_outputs: Union[
        RoutingModuleOutput,
        Sequence[Union[RoutingModuleOutput, Sequence[RoutingModuleOutput]]],
    ],
    N: Union[float, Sequence[float]] = 2.0,
    B: Optional[Union[int, Sequence[int]]] = None,
    gamma: float = 0.001,
    update_rule: str = "sign",
    max_bias: Optional[float] = 1.0,
    process_group: Optional[Any] = None,
    mask: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """
    Auxiliary-Loss-Free Load Balancing (DeepSeek MoE style, arXiv:2408.15664 / DeepSeek-V3).
    Nudges router boundary probabilities transparently by adjusting an expert/backbone-wise
    bias buffer without computing or backpropagating any gradients.

    In distributed / DDP training, performs an all-reduce across all ranks so that the global
    batch activation ratio is computed and router biases stay strictly synchronized across ranks.

    Correctly supports multi-stage and multi-backbone scenarios:
    Calculates target_ratio = 1 / (N_s * B_s) per stage s rather than conflating total routers
    with backbone count.

    Args:
        router_outputs: Flat or nested sequence of RoutingModuleOutput.
        N: Sparsity ratio per stage (single float or list of floats).
        B: Number of backbones per stage (single int, list of ints, or None to infer).
        gamma: Bias update step size (default: 0.001).
        update_rule: 'sign' (default, DeepSeek-V3 sign-based update), 'linear', 'proportional', or 'relative'.
        max_bias: Maximum absolute value for clamping biases (default: 1.0, None to disable).
        process_group: Optional torch.distributed ProcessGroup for distributed all-reduce.
        mask: Optional boolean tensor indicating valid tokens.

    Returns:
        Dictionary containing detailed statistics of the update.
    """
    if gamma <= 0.0:
        return {}

    stage_groups = _group_router_outputs_by_stage(router_outputs, B=B)
    if not stage_groups:
        return {}

    is_distributed = dist.is_available() and dist.is_initialized()
    stats: Dict[str, Any] = {}

    with torch.no_grad():
        for stage_idx in sorted(stage_groups.keys()):
            r_list = stage_groups[stage_idx]
            if len(r_list) == 0:
                continue

            n_val = (
                N[stage_idx]
                if isinstance(N, (list, tuple)) and stage_idx < len(N)
                else (N[-1] if isinstance(N, (list, tuple)) and len(N) > 0 else float(N))
            )
            first_r = r_list[0]
            b_val = (
                B[stage_idx]
                if isinstance(B, (list, tuple)) and stage_idx < len(B)
                else (
                    B
                    if isinstance(B, int)
                    else getattr(
                        first_r,
                        "n_backbones",
                        getattr(first_r.router_module, "n_backbones", len(r_list))
                        if first_r.router_module is not None
                        else len(r_list),
                    )
                )
            )

            target_ratio = 1.0 / float(n_val * b_val)

            for idx, r_out in enumerate(r_list):
                if not isinstance(r_out, RoutingModuleOutput):
                    continue

                boundary_mask = r_out.boundary_mask
                if boundary_mask.numel() == 0:
                    continue

                effective_mask = None
                if mask is not None and mask.shape == boundary_mask.shape:
                    effective_mask = mask.bool()
                elif getattr(r_out, "mask", None) is not None and r_out.mask.shape == boundary_mask.shape:
                    effective_mask = r_out.mask.bool()

                if effective_mask is not None:
                    local_active = boundary_mask[effective_mask].float().sum()
                    local_total = torch.tensor(
                        effective_mask.sum().item(),
                        device=boundary_mask.device,
                        dtype=torch.float32,
                    )
                else:
                    local_active = boundary_mask.float().sum()
                    local_total = torch.tensor(
                        boundary_mask.numel(),
                        device=boundary_mask.device,
                        dtype=torch.float32,
                    )

                # Distributed all-reduce over global batch
                if is_distributed:
                    reduce_tensor = torch.stack([local_active, local_total])
                    dist.all_reduce(
                        reduce_tensor,
                        op=dist.ReduceOp.SUM,
                        group=process_group,
                    )
                    global_active = reduce_tensor[0].item()
                    global_total = max(reduce_tensor[1].item(), 1.0)
                else:
                    global_active = local_active.item()
                    global_total = max(local_total.item(), 1.0)

                actual_ratio = global_active / global_total
                deviation = target_ratio - actual_ratio

                # Calculate bias update delta
                if update_rule == "sign":
                    if abs(deviation) > 1e-6:
                        delta = gamma * (1.0 if deviation > 0.0 else -1.0)
                    else:
                        delta = 0.0
                elif update_rule in ("linear", "proportional"):
                    delta = gamma * deviation
                elif update_rule == "relative":
                    delta = gamma * (deviation / max(target_ratio, 1e-6))
                else:
                    raise ValueError(f"Unknown update_rule: {update_rule}")

                # Apply to router_module bias buffer
                router_name = f"stage_{stage_idx}_router_{idx}"
                bias_val = 0.0
                if r_out.router_module is not None and hasattr(r_out.router_module, "bias"):
                    r_module = r_out.router_module
                    new_bias = r_module.bias + delta
                    if max_bias is not None:
                        new_bias = torch.clamp(new_bias, -max_bias, max_bias)
                    r_module.bias.copy_(new_bias)
                    bias_val = r_module.bias.item()

                stats[router_name] = {
                    "stage_idx": stage_idx,
                    "backbone_idx": getattr(r_out, "backbone_idx", idx),
                    "target_ratio": target_ratio,
                    "actual_ratio": actual_ratio,
                    "deviation": deviation,
                    "delta": delta,
                    "bias": bias_val,
                }

    return stats


class LossFreeLoadBalancer:
    """
    Auxiliary-Loss-Free Load Balancing Manager based on DeepSeek MoE (arXiv:2408.15664).

    Manages transparent, gradient-free nudging of boundary routing probabilities towards
    the stage-specific target sparsity ratio 1 / (N_s * B_s).

    Usage:
        balancer = LossFreeLoadBalancer(model=model, N=2.0, gamma=0.001)
        ...
        out = model(input_ids)
        loss = criterion(out.logits, labels)
        loss.backward()
        optimizer.step()
        balancer.step(out.bpred_output)  # Updates router biases with zero gradients
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        N: Optional[Union[float, Sequence[float]]] = None,
        B: Optional[Union[int, Sequence[int]]] = None,
        gamma: float = 0.001,
        update_rule: str = "sign",
        max_bias: Optional[float] = 1.0,
        process_group: Optional[Any] = None,
    ) -> None:
        self.model = model
        self.gamma = float(gamma)
        self.update_rule = update_rule
        self.max_bias = max_bias
        self.process_group = process_group

        eff_n = N
        eff_b = B
        # Auto-infer N, B, gamma, update_rule, max_bias from model config if model is provided
        if self.model is not None and hasattr(self.model, "config"):
            cfg = self.model.config
            if eff_b is None and hasattr(cfg, "num_backbones"):
                eff_b = cfg.num_backbones
            if eff_n is None and hasattr(cfg, "N"):
                eff_n = cfg.N
            if hasattr(cfg, "gamma") and gamma == 0.001:
                self.gamma = float(cfg.gamma)
            if hasattr(cfg, "update_rule") and update_rule == "sign":
                self.update_rule = str(cfg.update_rule)
            if hasattr(cfg, "max_bias") and max_bias == 1.0:
                self.max_bias = cfg.max_bias

        self.N = eff_n if eff_n is not None else 2.0
        self.B = eff_b

    def step(
        self,
        router_outputs: Union[
            RoutingModuleOutput,
            Sequence[Union[RoutingModuleOutput, Sequence[RoutingModuleOutput]]],
        ],
        N: Optional[Union[float, Sequence[float]]] = None,
        B: Optional[Union[int, Sequence[int]]] = None,
        gamma: Optional[float] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Executes one load-balancing update step over the batch router outputs.
        """
        eff_n = N if N is not None else self.N
        eff_b = B if B is not None else self.B
        eff_gamma = gamma if gamma is not None else self.gamma

        return update_loss_free_bias(
            router_outputs=router_outputs,
            N=eff_n,
            B=eff_b,
            gamma=eff_gamma,
            update_rule=self.update_rule,
            max_bias=self.max_bias,
            process_group=self.process_group,
            mask=mask,
        )

    def set_gamma(self, gamma: float) -> None:
        """Sets the bias update step size (useful for learning rate / gamma decay schedules)."""
        self.gamma = float(gamma)

    def reset_biases(self) -> None:
        """Resets all router biases in the model to 0."""
        if self.model is not None:
            for module in self.model.modules():
                if isinstance(module, RoutingModule):
                    module.reset_bias()

    def get_biases(self) -> Dict[str, float]:
        """Returns a dict of all current router biases in the model."""
        biases: Dict[str, float] = {}
        if self.model is not None:
            for name, module in self.model.named_modules():
                if isinstance(module, RoutingModule):
                    biases[name] = module.bias.item()
        return biases

    def set_biases(self, biases: Dict[str, float]) -> None:
        """Sets router biases from a dictionary mapping module names to bias values."""
        if self.model is not None:
            for name, module in self.model.named_modules():
                if isinstance(module, RoutingModule) and name in biases:
                    module.set_bias(biases[name])


def group_params(
    model: nn.Module,
    weight_decay: float = 0.0,
    lr: Optional[float] = None,
    lr_multiplier: Optional[Sequence[float]] = None,
    no_decay_keywords: Sequence[str] = ("bias", "norm", "pad_dimension", "rmsnorm", "layernorm"),
    trainable_only: bool = True,
) -> List[Dict[str, Any]]:
    """
    Creates parameter groups for the optimizer based on learning rate multipliers and weight decay.

    Safely handles parameters without '_optim' defined yet, preventing AttributeError crashes.
    Applies weight decay = 0.0 to 1D parameters, biases, norms, and specified no-decay parameters.
    Merges parameters with identical optimizer configurations into distinct parameter groups.

    Args:
        model: The model to group parameters for.
        weight_decay: Default weight decay to apply to matrix/tensor parameters (default: 0.0).
        lr: Optional base learning rate to assign to the groups.
        lr_multiplier: Optional sequence of learning rate multipliers per hierarchy stage
                       (outer stage first, e.g. [3.0, 1.7, 0.9]). If provided and model
                       implements `apply_lr_multiplier`, it will be invoked automatically.
        no_decay_keywords: Parameter name substrings that should have weight_decay=0.0.
        trainable_only: If True, only parameters with requires_grad=True are included (default: True).

    Returns:
        A list of parameter group dicts compatible with PyTorch optimizers:
        [
            {"params": [param1, ...], "weight_decay": float, "lr": float, ...},
            ...
        ]
    """
    # If lr_multiplier is provided, apply it to the model parameters if supported
    if lr_multiplier is not None and hasattr(model, "apply_lr_multiplier"):
        model.apply_lr_multiplier(list(lr_multiplier))

    # Safely ensure all parameters have appropriate _optim annotations
    for name, param in model.named_parameters():
        if trainable_only and not param.requires_grad:
            continue

        is_no_decay = (param.ndim <= 1) or any(
            kw.lower() in name.lower() for kw in no_decay_keywords
        )

        if not hasattr(param, "_optim"):
            param._optim = {}

        if is_no_decay:
            if "weight_decay" not in param._optim:
                apply_optimization_params(param, weight_decay=0.0)
        else:
            if "weight_decay" not in param._optim and weight_decay > 0.0:
                apply_optimization_params(param, weight_decay=weight_decay)

        if lr is not None:
            if "lr" not in param._optim:
                mult = param._optim.get("lr_multiplier", 1.0)
                apply_optimization_params(param, lr=float(lr * mult))

    # Collect all unique optimization keys
    all_keys = set()
    for param in model.parameters():
        if trainable_only and not param.requires_grad:
            continue
        optim_dict = getattr(param, "_optim", {})
        all_keys.update(optim_dict.keys())

    sorted_keys = sorted(list(all_keys))
    groups_map: Dict[tuple, Dict[str, Any]] = {}

    for name, param in model.named_parameters():
        if trainable_only and not param.requires_grad:
            continue

        optim_dict = getattr(param, "_optim", {})
        group_key = tuple((k, optim_dict.get(k, None)) for k in sorted_keys)

        if group_key not in groups_map:
            entry: Dict[str, Any] = {"params": [param]}
            for k in sorted_keys:
                val = optim_dict.get(k, None)
                if val is not None:
                    entry[k] = val
            groups_map[group_key] = entry
        else:
            groups_map[group_key]["params"].append(param)

    return list(groups_map.values())


