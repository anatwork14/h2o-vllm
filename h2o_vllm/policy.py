"""CPU-only reference H2O heavy-hitter and recent-token selection policy.

Provenance: FMInference/H2O@ac75c2a8a9e76832b2a4139b9363373b56336bfb
h2o_hf/utils_real_drop/modify_llama.py::H2OKVCache_LayerWise.

NOTE: This code does not capture real attention scores or physically evict
vLLM KV blocks. record_eviction() only records the result of an eviction
that an external owner has ALREADY committed, so dropped tokens cannot
magically re-enter candidate sets.
"""
from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from numbers import Real


@dataclass(frozen=True)
class H2OPolicyConfig:
    heavy_hitter_tokens: int
    recent_tokens: int
    sink_tokens: int = 0
    block_size: int = 16

    def __post_init__(self) -> None:
        for name in ("heavy_hitter_tokens", "recent_tokens", "sink_tokens", "block_size"):
            val = getattr(self, name)
            if isinstance(val, bool) or not isinstance(val, int):
                raise TypeError(f"{name} must be an integer")
            if val < 0 or (name == "block_size" and val == 0):
                raise ValueError(f"{name} must be nonnegative; block_size positive")
        if self.budget_tokens == 0:
            raise ValueError("At least one cache segment must be retained")

    @property
    def budget_tokens(self) -> int:
        return self.heavy_hitter_tokens + self.recent_tokens + self.sink_tokens


@dataclass
class H2OPolicy:
    """One request/layer/head scoring stream with absolute token positions.

    Attention mass must come from genuinely observed, normalized attention
    probabilities (possibly summed across query positions) and not be inferred
    from outputs of fused attention kernels.
    """
    config: H2OPolicyConfig
    scores: dict[int, float] = field(default_factory=dict)
    _last_length: int = 0
    _evicted: set[int] = field(default_factory=set)

    def observe(self, sequence_length: int, attention_mass: Mapping[int, float]) -> None:
        if isinstance(sequence_length, bool) or not isinstance(sequence_length, int):
            raise TypeError("sequence_length must be an integer")
        if sequence_length < self._last_length:
            raise ValueError("Sequence rewound; restore a consistent checkpoint")
        if sequence_length < 0:
            raise ValueError("sequence_length cannot be negative")
        updates: dict[int, float] = {}
        for pos, mass in attention_mass.items():
            if (not isinstance(pos, int) or isinstance(pos, bool)
                or pos < 0 or pos >= sequence_length or pos in self._evicted):
                raise ValueError(f"Invalid/unavailable logical position: {pos}")
            if isinstance(mass, bool) or not isinstance(mass, Real):
                raise TypeError("Attention mass must be a real numeric value")
            v = float(mass)
            if not isfinite(v) or v < 0:
                raise ValueError("Attention mass must be finite and nonnegative")
            updated = self.scores.get(pos, 0.0) + v
            if not isfinite(updated):
                raise ValueError("Accumulated attention score overflow")
            updates[pos] = updated
        # Atomic: neither sequence length nor scores are modified on error.
        self.scores.update(updates)
        self._last_length = sequence_length

    def kept_positions(self, sequence_length: int) -> tuple[int, ...]:
        if isinstance(sequence_length, bool) or not isinstance(sequence_length, int):
            raise TypeError("sequence_length must be an integer")
        if sequence_length < 0:
            raise ValueError("sequence_length cannot be negative")
        if sequence_length < self._last_length:
            raise ValueError("Cannot select a sequence earlier than observed state")
        if sequence_length == 0:
            return ()
        available = set(range(sequence_length)) - self._evicted
        if len(available) <= self.config.budget_tokens:
            return tuple(sorted(available))
        cfg = self.config
        sink = set(range(min(cfg.sink_tokens, sequence_length)))
        recent = set(range(max(0, sequence_length - cfg.recent_tokens), sequence_length))
        if not (sink | recent) <= available:
            raise RuntimeError("Already-evicted token entered mandatory sink/recent set")
        protected = sink | recent
        candidates = available - protected
        heavy = sorted(candidates, key=lambda p: (-self.scores.get(p, 0.0), p))[:cfg.heavy_hitter_tokens]
        return tuple(sorted(protected | set(heavy)))

    def evict_positions(self, sequence_length: int) -> tuple[int, ...]:
        keep = set(self.kept_positions(sequence_length))
        return tuple(pos for pos in range(sequence_length) if pos not in keep and pos not in self._evicted)

    def record_eviction(self, sequence_length: int, physically_evicted: Mapping[int, object] | set[int]) -> None:
        """Record *completed* external physical evictions, never perform them.

        Rejects attempted removal of protected tokens or unknown/old positions.
        This does not synchronize with vLLM; an engine integration must do that.
        """
        proposed = set(physically_evicted)
        eligible = set(self.evict_positions(sequence_length))
        if not proposed <= eligible:
            raise ValueError("Physical eviction record conflicts with H2O keep decision")
        self._evicted.update(proposed)
        for pos in proposed:
            self.scores.pop(pos, None)

    def kept_blocks(self, sequence_length: int) -> tuple[int, ...]:
        return tuple(sorted({p // self.config.block_size for p in self.kept_positions(sequence_length)}))

    def reset(self) -> None:
        self.scores.clear()
        self._evicted.clear()
        self._last_length = 0
