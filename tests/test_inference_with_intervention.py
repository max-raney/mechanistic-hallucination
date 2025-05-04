import pytest
import torch
from mechanistic_interventions.scripts.inference_with_intervention import load_model_and_tokenizer, alpha_hook


def test_load_model_and_tokenizer():
    device = "cpu"
    model_path = "google/gemma-2b"
    model, tokenizer = load_model_and_tokenizer(model_path, device)
    assert model is not None
    assert tokenizer is not None


def test_alpha_hook():
    direction = torch.randn(10)
    alpha = 1.0
    mode = "suppress"
    hook = alpha_hook(direction, alpha, mode)
    assert callable(hook) 