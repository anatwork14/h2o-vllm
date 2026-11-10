"""Reference H2O heavy-hitter + recent-token cache policy.

Ported from the idea and selection semantics of FMInference/H2O at
ac75c2a8a9e76832b2a4139b9363373b56336bfb, in particular
h2o_hf/utils_real_drop/modify_llama.py::H2OKVCache_LayerWise.

This pure-Python selector accepts *observed attention probabilities*. It does
not manufacture attention scores, infer them from attention output, alter vLLM
metadata, or free vLLM KV blocks. A production adapter must supply real
per-layer attention probabilities and make physical evictions atomically with
vLLM's scheduler/cache manager.
"""
from dataclasses import dataclass, field
from collections.abc import Mapping
from math import isfinite

@dataclass(frozen=True)
class H2OPolicyConfig:
    heavy_hitter_tokens: int
    recent_tokens: int
    sink_tokens: int = 0
    block_size: int = 16

    def __post_init__(self) -> None:
        for name in ("heavy_hitter_tokens", "recent_tokens", "sink_tokens"):
            if not isinstance(getattr(self, name), int) or isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be an integer")
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.heavy_hitter_tokens + self.recent_tokens + self.sink_tokens == 0:
            raise ValueError("At least one cache segment must be retained")
        if not isinstance(self.block_size, int) or self.block_size <= 0:
            raise ValueError("block_size must be a positive integer")

    @property
    def budget_tokens(self) -> int:
        return self.heavy_hitter_tokens + self.recent_tokens + self.sink_tokens

@dataclass
class H2OPolicy:
    """One request and one layer or an explicitly aggregated layer group.

    Positions are absolute logical token positions, never compacted indices.
    observe() expects actual normalized post-softmax attention probability mass.
    """
    config: H2OPolicyConfig
    scores: dict[int, float] = field(default_factory=dict)
    _last_length: int = 0

    def observe(self, sequence_length: int, attention_mass: Mapping[int, float]) -> None:
        if sequence_length < self._last_length:
            raise ValueError("Sequence rewound: start a new request or restore a checkpoint")
        if sequence_length < 0:
            raise ValueError("sequence_length cannot be negative")
        updates: dict[int, float] = {}
        for pos, mass in attention_mass.items():
            if not isinstance(pos, int) or isinstance(pos, bool) or pos < 0 or pos >= sequence_length:
                raise ValueError(f"Invalid logical token position: {pos}")
            if not isfinite(float(mass)) or mass < 0:
                raise ValueError("Attention masses must be finite and nonnegative")
            updates[pos] = float(mass)
        for pos, mass in updates.items():
            self.scores[pos] = self.scores.get(pos, 0.0) + mass
        self._last_length = sequence_length

    def kept_positions(self, sequence_length: int) -> tuple[int, ...]:
        if sequence_length < 0:
            raise ValueError("sequence_length cannot be negative")
        if sequence_length <= self.config.budget_tokens:
            return tuple(range(sequence_length))
        cfg = self.config
        sink = set(range(min(cfg.sink_tokens, sequence_length)))
        recent = set(range(max(0, sequence_length - cfg.recent_tokens), sequence_length))
        protected = sink | recent
        candidates = (pos for pos in range(sequence_length) if pos not in protected)
        heavy = sorted(candidates, key=lambda pos: (-self.scores.get(pos, 0.0), pos))[:cfg.heavy_hitter_tokens]
        return tuple(sorted(protected | set(heavy)))

    def evict_positions(self, sequence_length: int) -> tuple[int, ...]:
        keep = set(self.kept_positions(sequence_length))
        return tuple(pos for pos in range(sequence_length) if pos not in keep)

    def kept_blocks(self, sequence_length: int) -> tuple[int, ...]:
        # Selection proposal ONLY, not vLLM block-table compaction or physical KV eviction.
        return tuple(sorted({pos // self.config.block_size for pos in self.kept_positions(sequence_length)}))

    def reset(self) -> None:
        self.scores.clear()
        self._last_length = 0
