import os
import pytest

def test_hf_token_loaded(monkeypatch):
    # simulate a .env value
    monkeypatch.setenv("HF_TOKEN", "hf_dummy")
    # reload the module to pick up the monkeypatched env
    import importlib
    cfg = importlib.reload(__import__("mechanistic_interventions.config", fromlist=[""]))
    assert cfg.HF_TOKEN == "hf_dummy"

def test_missing_token_raises(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    import importlib
    with pytest.raises(RuntimeError):
        importlib.reload(__import__("mechanistic_interventions.config", fromlist=[""]))
