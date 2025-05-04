import pytest
import torch
import numpy as np
from mechanistic_interventions.sandboxs.sandbox_gemma import (
    load_gemma_model,
    scan_all_layers,
    load_prompts
)


@pytest.fixture
def mock_model():
    class MockModel:
        def __init__(self):
            self.model = MockModelLayers()
            self.config = type('Config', (), {'num_hidden_layers': 10})
            
        def to(self, device):
            return self
            
        def eval(self):
            return self
            
        def __call__(self, **kwargs):
            # Simulate forward pass through all layers
            for layer in self.model.layers:
                layer()  # This will trigger the hook and store activations
            return torch.randn(1, 5, 10)  # batch_size=1, seq_len=5, hidden_size=10
            
    class MockModelLayers:
        def __init__(self):
            self.layers = [MockLayer() for _ in range(10)]
            
    class MockLayer:
        def __init__(self):
            self.mlp = MockMLP()
            self.hidden_states = []  # Store activations here
            self.hook_fn = None
            
        def register_forward_hook(self, hook_fn):
            self.hook_fn = hook_fn
            return MockHook()  # Return a proper hook object
            
        def __call__(self, *args, **kwargs):
            # Generate dummy output and store it
            output = torch.randn(1, 5, 10)  # Dummy output
            if self.hook_fn:
                self.hook_fn(self, None, output)
            self.hidden_states.append(output)
            return output
            
    class MockMLP:
        def __init__(self):
            self.hidden_states = []  # Store activations here
            self.hook_fn = None
            
        def register_forward_hook(self, hook_fn):
            self.hook_fn = hook_fn
            return MockHook()  # Return a proper hook object
            
        def __call__(self, *args, **kwargs):
            # Generate dummy output and store it
            output = torch.randn(1, 5, 10)  # Dummy output
            if self.hook_fn:
                self.hook_fn(self, None, output)
            self.hidden_states.append(output)
            return output
            
    class MockHook:
        def remove(self):
            pass  # No-op for mock hook
            
    return MockModel()


@pytest.fixture
def mock_tokenizer():
    class MockTokenizer:
        def __init__(self):
            pass
            
        def __call__(self, text, return_tensors=None, **kwargs):
            # Return a dictionary with tensors that can be moved to a device
            return {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]])
            }
            
    return MockTokenizer()


def test_load_gemma_model(mock_model, mock_tokenizer, monkeypatch):
    # Mock the model loading
    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM.from_pretrained",
        lambda *args, **kwargs: mock_model
    )
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: mock_tokenizer
    )
    
    model, tokenizer = load_gemma_model()
    assert model is not None
    assert tokenizer is not None


def test_scan_all_layers(mock_model, mock_tokenizer):
    # Create a dummy direction vector
    direction_vector = torch.randn(10)
    
    # Test without direction vector
    scores = scan_all_layers(mock_model, mock_tokenizer, "test prompt")
    assert len(scores) == mock_model.config.num_hidden_layers
    assert all(isinstance(score, float) for score in scores)
    
    # Test with direction vector
    scores = scan_all_layers(mock_model, mock_tokenizer, "test prompt", direction_vector)
    assert len(scores) == mock_model.config.num_hidden_layers
    assert all(isinstance(score, float) for score in scores)


def test_load_prompts(tmp_path):
    # Create a temporary prompt file
    prompt_file = tmp_path / "prompts.txt"
    with open(prompt_file, "w") as f:
        f.write("prompt1\nprompt2\nprompt3\n")
    
    prompts = load_prompts(str(prompt_file))
    assert len(prompts) == 3
    assert prompts[0] == "prompt1"
    assert prompts[1] == "prompt2"
    assert prompts[2] == "prompt3" 