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
    def generate(self, **kwargs):
        import torch
        # simulate 1 input token + 2 generated
        return torch.tensor([[0, 1, 2]])

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
    res = benchmark_model("dummy/model", "hi", token="fake", device=None)
    # confirm fallback to cpu
    assert res["model_id"] == "dummy/model"
    for key in (
        "load_time_sec", "inference_time_sec", 
        "peak_memory_gb", "tokens_per_sec", "output_text"
    ):
        assert key in res
    # on CPU, peak_memory should be zero
    assert res["peak_memory_gb"] == 0.0
    # output_text comes from DummyTok.decode
    assert res["output_text"] == "dummy output"
