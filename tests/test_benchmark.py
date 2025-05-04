import pytest
import torch

from mechanistic_interventions.evaluation.benchmark import benchmark_model

class DummyTok:
    def __call__(self, txt, return_tensors):
        import torch
        return {"input_ids": torch.tensor([[0]])}
    def decode(self, *args, **kwargs):
        return "dummy output"

class DummyModel:
    def __init__(self):
        self.to = lambda x: self  # Mock 'to' method to return self
        self.generate = lambda **kwargs: torch.tensor([[0, 1, 2]])  # Mock generate method

@pytest.fixture(autouse=True)
def stub_transformers(monkeypatch):
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *a, **k: DummyTok()
    )
    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM.from_pretrained",
        lambda *a, **k: DummyModel()
    )
    # force CPU path (no GPU available in CI)
    monkeypatch.setattr(torch, "cuda", type("C", (), {"is_available": staticmethod(lambda: False)}))

def test_benchmark_model_keys_and_types():
    res = benchmark_model("dummy/model", "hi", token="fake", device="cpu")
    assert isinstance(res, dict)
    assert "load_time_sec" in res
    assert "inference_time_sec" in res
    assert "peak_memory_gb" in res
    assert "tokens_per_sec" in res
    assert "output_text" in res
