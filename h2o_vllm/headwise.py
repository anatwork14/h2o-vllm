"""Per-head H2O selection parity reference; block union is not a KV-eviction engine."""
from dataclasses import dataclass, field
from collections.abc import Mapping, Sequence
from .policy import H2OPolicy, H2OPolicyConfig

@dataclass
class H2OHeadwisePolicy:
    """Original H2O selects heavy hitters per attention head.

    A vLLM shared-block union may exceed the per-head token budget greatly.
    This is NOT a memory-saving substitute for physical per-head KV eviction.
    """
    config: H2OPolicyConfig
    num_heads: int
    heads: list[H2OPolicy] = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.num_heads, int) or self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        self.heads = [H2OPolicy(self.config) for _ in range(self.num_heads)]

    def observe(self, seq_len: int, per_head_mass: Sequence[Mapping[int, float]]) -> None:
        if len(per_head_mass) != self.num_heads:
            raise ValueError("Per-head scores must have one entry per head")
        shadow = [H2OPolicy(self.config, dict(p.scores), p._last_length) for p in self.heads]
        for policy, mass in zip(shadow, per_head_mass):
            policy.observe(seq_len, mass)
        self.heads = shadow

    def kept_by_head(self, seq_len: int) -> tuple[tuple[int, ...], ...]:
        return tuple(head.kept_positions(seq_len) for head in self.heads)

    def union_blocks(self, seq_len: int) -> tuple[int, ...]:
        return tuple(sorted({b for head in self.heads for b in head.kept_blocks(seq_len)}))
