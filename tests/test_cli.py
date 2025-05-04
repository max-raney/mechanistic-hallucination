import pytest
import torch
from mechanistic_interventions.evaluation.benchmark import benchmark_model

@pytest.fixture(autouse=True)
def mock_transformers(monkeypatch):
    class DummyTok:
        def __call__(self, txt, return_tensors):
            return {"input_ids": torch.tensor([[0]])}
        def decode(self, *args, **kwargs):
            return "dummy output"
            
    class DummyModel:
        def __init__(self):
            self.to = lambda x: self
            self.generate = lambda **kwargs: torch.tensor([[0, 1, 2]])
            
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

def test_benchmark_model():
    result = benchmark_model("dummy/model", "test prompt", device="cpu")
    assert isinstance(result, dict)
    assert "model_id" in result
    assert "load_time_sec" in result
    assert "inference_time_sec" in result
    assert "peak_memory_gb" in result
    assert "tokens_per_sec" in result
    assert "output_text" in result
