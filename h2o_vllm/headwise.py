"""Per-query-head H2O selection reference: NOT a vLLM eviction backend."""
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from .policy import H2OPolicy, H2OPolicyConfig


@dataclass
class H2OHeadwisePolicy:
    """Head-dependent selections, potentially far above shared-block budgets.

    In grouped-query attention (GQA), query heads and KV heads differ; this
    model has NOT reconciled head-dependent selections into a real GQA kernel.
    """
    config: H2OPolicyConfig
    num_heads: int
    heads: list[H2OPolicy] = field(init=False)

    def __post_init__(self) -> None:
        if isinstance(self.num_heads, bool) or not isinstance(self.num_heads, int) or self.num_heads <= 0:
            raise ValueError("num_heads must be a positive integer")
        self.heads = [H2OPolicy(self.config) for _ in range(self.num_heads)]

    def observe(self, seq_len: int, per_head_mass: Sequence[Mapping[int, float]]) -> None:
        if len(per_head_mass) != self.num_heads:
            raise ValueError("Exactly one mass mapping per query head is required")
        shadow = [
            H2OPolicy(p.config, dict(p.scores), p._last_length, set(p._evicted))
            for p in self.heads
        ]
        for policy, mass in zip(shadow, per_head_mass):
            policy.observe(seq_len, mass)
        self.heads = shadow

    def kept_by_head(self, seq_len: int) -> tuple[tuple[int, ...], ...]:
        return tuple(p.kept_positions(seq_len) for p in self.heads)

    def union_blocks(self, seq_len: int) -> tuple[int, ...]:
        return tuple(sorted({b for p in self.heads for b in p.kept_blocks(seq_len)}))
