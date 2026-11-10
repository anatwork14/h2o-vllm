"""Regression coverage added after independent integration review."""
from unittest.mock import patch
import sys
import types

import pytest

from h2o_vllm.policy import H2OPolicy, H2OPolicyConfig
from h2o_vllm.headwise import H2OHeadwisePolicy
from h2o_vllm.benchmark_contract import manifest
from h2o_vllm.vllm_plugin import register


def test_disabled_plugin_does_not_break_unrelated_engines(monkeypatch):
    monkeypatch.delenv("H2O_VLLM_MODE", raising=False)
    with patch("h2o_vllm.vllm_plugin.assert_engine_version", side_effect=AssertionError("wrong")):
        register()


def test_passthrough_plugin_registers_exact_backend(monkeypatch):
    monkeypatch.setenv("H2O_VLLM_MODE", "passthrough")
    registry = types.ModuleType("vllm.v1.attention.backends.registry")
    registry.AttentionBackendEnum = types.SimpleNamespace(CUSTOM=object())
    called = []
    registry.register_backend = lambda enum, path: called.append((enum, path))
    # Stub the exact imported module without installing the 1+GB GPU engine.
    with patch("h2o_vllm.vllm_plugin.assert_engine_version", return_value="0.29.0"), patch.dict(
        sys.modules, {"vllm": types.ModuleType("vllm"),
                      "vllm.v1": types.ModuleType("vllm.v1"),
                      "vllm.v1.attention": types.ModuleType("vllm.v1.attention"),
                      "vllm.v1.attention.backends": types.ModuleType("vllm.v1.attention.backends"),
                      "vllm.v1.attention.backends.registry": registry}
    ):
        register()
    assert called == [(registry.AttentionBackendEnum.CUSTOM,
                       "h2o_vllm.vllm_backend.H2OPassthroughBackend")]


def test_no_unsupported_mode_bypasses_guard(monkeypatch):
    monkeypatch.setenv("H2O_VLLM_MODE", "h2o")
    with pytest.raises(RuntimeError, match="not implemented"):
        register()


def test_overflow_is_atomic():
    p = H2OPolicy(H2OPolicyConfig(1, 1))
    p.observe(10, {1: 1e308})
    with pytest.raises(ValueError, match="overflow"):
        p.observe(11, {1: 1e308, 2: 1.0})
    assert p.scores == {1: 1e308}
    assert p._last_length == 10


def test_record_eviction_is_irreversible():
    p = H2OPolicy(H2OPolicyConfig(1, 2))
    p.observe(8, {2: 0.9, 1: 0.1})
    drop = set(p.evict_positions(8))
    assert 2 in p.kept_positions(8)
    p.record_eviction(8, drop)
    p.observe(9, {2: 0.5, 7: 0.3})  # scores for retained positions only
    assert not (drop & set(p.kept_positions(9)))
    with pytest.raises(ValueError):
        p.observe(10, {min(drop): 0.01})
    with pytest.raises(ValueError):
        p.record_eviction(9, {2})


def test_headwise_rollback_with_eviction_state():
    x = H2OHeadwisePolicy(H2OPolicyConfig(1, 1), 2)
    x.observe(6, [{1: 0.9}, {2: 0.8}])
    x.heads[0].record_eviction(6, set(x.heads[0].evict_positions(6)))
    state = (dict(x.heads[0].scores), set(x.heads[0]._evicted))
    with pytest.raises(ValueError):
        x.observe(7, [{1: 0.1}, {8: 0.2}])
    assert (x.heads[0].scores, x.heads[0]._evicted) == state


def test_manifest_makes_runtime_false_explicit():
    m = manifest(H2OPolicyConfig(1, 1))
    assert m["runtime_verified"] is False
    assert m["physical_kv_eviction"] is False
    assert m["attention_probabilities_collected"] is False
