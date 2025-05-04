import pytest
import torch
from mechanistic_interventions.scripts.mean_diff import load_model_and_tokenizer, compute_mean_diff


def test_load_model_and_tokenizer():
    device = "cpu"
    model_path = "google/gemma-2b"
    model, tokenizer = load_model_and_tokenizer(model_path, device)
    assert model is not None
    assert tokenizer is not None


def test_compute_mean_diff():
    device = "cpu"
    model_path = "google/gemma-2b"
    model, tokenizer = load_model_and_tokenizer(model_path, device)
    prompts = ["prompt1", "prompt2"]
    labels = ["label1", "label2"]
    compute_mean_diff(model, tokenizer, prompts, labels, device)
    # Add assertions based on expected behavior 