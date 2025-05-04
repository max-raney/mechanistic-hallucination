import pytest
import torch
from mechanistic_interventions.scripts.train_concept_vectors import load_model_and_tokenizer, train_concept_vectors


def test_load_model_and_tokenizer():
    device = "cpu"
    model_path = "google/gemma-2b"
    model, tokenizer = load_model_and_tokenizer(model_path, device)
    assert model is not None
    assert tokenizer is not None


def test_train_concept_vectors():
    device = "cpu"
    model_path = "google/gemma-2b"
    model, tokenizer = load_model_and_tokenizer(model_path, device)
    prompts = ["prompt1", "prompt2"]
    labels = ["label1", "label2"]
    train_concept_vectors(model, tokenizer, prompts, labels, device)
    # Add assertions based on expected behavior 