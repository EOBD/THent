from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union


@dataclass
class AttnConfig:

    num_heads: List = field(default_factory=list)
    rotary_emb_dim: List = field(default_factory=list)
    window_size: List = field(default_factory=list)


@dataclass
class SSMConfig:

    d_conv: int = 4
    expand: int = 2
    d_state: int = 128
    chunk_size: int = 256


@dataclass
class HNetConfig:
    arch_layout: List[Union[str, List]] = field(default_factory=list)
    d_model: List[int] = field(default_factory=list)
    # intermediate dimension for the FFNs (0 indicates no FFN)
    d_intermediate: List[int] = field(default_factory=list)
    vocab_size: int = 256
    ssm_cfg: SSMConfig = field(default_factory=SSMConfig)
    attn_cfg: AttnConfig = field(default_factory=AttnConfig)
    tie_embeddings: bool = False
    num_backbones: Union[int, List[int]] = 1
    join_mode: Union[str, List[str]] = "independent"
    # Target downsampling / compression ratio per stage (e.g. 2.0 or [2.0, 4.0])
    N: Union[float, List[float]] = 2.0
    gamma: float = 0.001
    max_bias: Optional[float] = 1.0
    update_rule: str = "sign"

    def get_num_backbones(self, stage_idx: int = 0) -> int:
        """Returns the number of backbones for a specific stage."""
        if isinstance(self.num_backbones, (list, tuple)):
            if stage_idx < len(self.num_backbones):
                return int(self.num_backbones[stage_idx])
            return int(self.num_backbones[-1]) if len(self.num_backbones) > 0 else 1
        return int(self.num_backbones)

    def get_target_compression(self, stage_idx: int = 0) -> float:
        """Returns the target downsampling ratio N for a specific stage."""
        if isinstance(self.N, (list, tuple)):
            if stage_idx < len(self.N):
                return float(self.N[stage_idx])
            return float(self.N[-1]) if len(self.N) > 0 else 2.0
        return float(self.N)

    def get_num_stages(self) -> int:
        """Determines the maximum number of hierarchical stages from arch_layout."""
        if not self.arch_layout:
            return 1

        def _count_stages(layout) -> int:
            if isinstance(layout, list):
                if len(layout) == 3 and isinstance(layout[0], str) and isinstance(layout[2], str):
                    return 1 + _count_stages(layout[1])
                elif len(layout) > 0 and all(isinstance(item, (list, str)) for item in layout):
                    return max((_count_stages(item) for item in layout), default=1)
            return 0

        total_stages = _count_stages(self.arch_layout)
        return max(total_stages, 1)

