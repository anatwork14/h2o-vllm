import pytest
from h2o_vllm.headwise import H2OHeadwisePolicy
from h2o_vllm.policy import H2OPolicyConfig

def test_independent_heads_can_require_more_shared_blocks():
    cfg = H2OPolicyConfig(1, 1, block_size=2)
    obj = H2OHeadwisePolicy(cfg, 2)
    obj.observe(10, [{1: .9}, {5: .9}])
    assert obj.kept_by_head(10) == ((1, 9), (5, 9))
    assert obj.union_blocks(10) == (0, 2, 4)

def test_headwise_batch_atomic_on_bad_mass():
    obj = H2OHeadwisePolicy(H2OPolicyConfig(1, 1), 2)
    with pytest.raises(ValueError):
        obj.observe(10, [{1: .5}, {3: float("nan")}])
    assert obj.heads[0].scores == {}
    assert obj.heads[1].scores == {}
