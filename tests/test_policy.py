import pytest
from h2o_vllm.policy import H2OPolicy, H2OPolicyConfig
from h2o_vllm.session import H2OSessions
from h2o_vllm.benchmark_contract import manifest

def test_h2o_keeps_heavy_and_recent_and_sink():
    cfg = H2OPolicyConfig(heavy_hitter_tokens=2, recent_tokens=3, sink_tokens=1, block_size=4)
    policy = H2OPolicy(cfg)
    policy.observe(12, {1: .1, 2: .9, 3: .8, 5: .2, 8: .3})
    assert policy.kept_positions(12) == (0, 2, 3, 9, 10, 11)
    assert policy.kept_blocks(12) == (0, 2)
    assert len(policy.evict_positions(12)) == 6

def test_incremental_scores_accumulate_and_ties_stable():
    policy = H2OPolicy(H2OPolicyConfig(1, 1))
    policy.observe(5, {1: 0.3, 2: 0.4})
    policy.observe(6, {1: 0.5})
    assert policy.kept_positions(6) == (1, 5)
    policy.reset()
    assert policy.kept_positions(6) == (0, 5)

def test_short_sequences_keep_everything():
    policy = H2OPolicy(H2OPolicyConfig(1, 2, sink_tokens=1))
    assert policy.kept_positions(4) == (0, 1, 2, 3)

def test_invalid_attention_fails_closed_without_partial_update():
    policy = H2OPolicy(H2OPolicyConfig(1, 1))
    with pytest.raises(ValueError):
        policy.observe(10, {1: 0.1, 3: float("nan")})
    assert policy.scores == {}
    with pytest.raises(ValueError):
        policy.observe(10, {10: 0.1})
    with pytest.raises(ValueError):
        policy.observe(10, {-1: 0.1})
    with pytest.raises(ValueError):
        policy.observe(10, {0: -0.1})

def test_rewind_rejected():
    policy = H2OPolicy(H2OPolicyConfig(1, 1))
    policy.observe(8, {})
    with pytest.raises(ValueError):
        policy.observe(7, {})

def test_request_layer_isolation_and_cleanup():
    sessions = H2OSessions(H2OPolicyConfig(1, 1))
    sessions.get("req-1", "layer-0").observe(8, {2: .7})
    assert sessions.get("req-1", "layer-1").scores == {}
    assert sessions.get("req-2", "layer-0").scores == {}
    sessions.finish_request("req-1")
    assert sessions.get("req-1", "layer-0").scores == {}

def test_benchmark_manifest_cannot_mislabel_passthrough():
    cfg = H2OPolicyConfig(4, 4)
    item = manifest(cfg, mode="passthrough")
    assert item["engine_version"] == "0.29.0"
    assert item["method"] != "h2o"
    assert not item["real_h2o_eviction"]
    with pytest.raises(ValueError):
        manifest(cfg, mode="h2o")

def test_config_validation():
    with pytest.raises(ValueError):
        H2OPolicyConfig(0, 0)
    with pytest.raises(ValueError):
        H2OPolicyConfig(1, -1)
    with pytest.raises(TypeError):
        H2OPolicyConfig(True, 1)
