import pytest
import contextlib

from mechanistic_interventions.utils import device as dev

@pytest.fixture(autouse=True)
def force_cpu_device(monkeypatch):
    # Always pick CPU in tests
    monkeypatch.setattr(dev, "get_default_device", lambda: "cpu")
    # Make autocast a no-op
    monkeypatch.setattr(dev, "autocast", contextlib.nullcontext)