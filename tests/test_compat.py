from unittest.mock import patch
import pytest
from h2o_vllm.compat import assert_engine_version

def test_engine_version_pinned():
    with patch("h2o_vllm.compat.metadata.version", return_value="0.29.0"):
        assert assert_engine_version() == "0.29.0"
    with patch("h2o_vllm.compat.metadata.version", return_value="0.29.1"):
        with pytest.raises(RuntimeError, match="requires vLLM"):
            assert_engine_version()

def test_plugin_blocks_fake_h2o(monkeypatch):
    with patch("h2o_vllm.vllm_plugin.assert_engine_version", return_value="0.29.0"):
        monkeypatch.setenv("H2O_VLLM_MODE", "h2o")
        from h2o_vllm.vllm_plugin import register
        with pytest.raises(RuntimeError, match="not integrated"):
            register()
