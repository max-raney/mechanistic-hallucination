import pytest
import torch
from mechanistic_interventions.scripts.inference_with_intervention import alpha_hook


@pytest.fixture
def dummy_data():
    direction = torch.randn(10)
    output = torch.randn(1, 5, 10)  # batch_size=1, seq_len=5, hidden_size=10
    return direction, output


def test_alpha_hook_suppress(dummy_data):
    direction, output = dummy_data
    alpha = 1.0
    
    # Test suppress mode
    hook = alpha_hook(direction, alpha, mode="suppress")
    modified_output = hook(None, None, output)
    
    # Check that the output was modified
    assert not torch.allclose(output, modified_output)
    # Check that the norm was reduced
    assert torch.norm(modified_output) < torch.norm(output)
    # Check that the projection was removed
    proj = torch.einsum("bsd,d->bs", output, direction)
    modified_proj = torch.einsum("bsd,d->bs", modified_output, direction)
    assert torch.norm(modified_proj) < torch.norm(proj)


def test_alpha_hook_enhance(dummy_data):
    direction, output = dummy_data
    alpha = 1.0
    
    # Test enhance mode
    hook = alpha_hook(direction, alpha, mode="enhance")
    modified_output = hook(None, None, output)
    
    # Check that the output was modified
    assert not torch.allclose(output, modified_output)
    # Check that the norm was increased
    assert torch.norm(modified_output) > torch.norm(output)
    # Check that the projection was enhanced
    proj = torch.einsum("bsd,d->bs", output, direction)
    modified_proj = torch.einsum("bsd,d->bs", modified_output, direction)
    assert torch.norm(modified_proj) > torch.norm(proj)


def test_alpha_hook_different_alphas(dummy_data):
    direction, output = dummy_data
    
    # Test different alpha values
    alphas = [0.1, 1.0, 10.0]
    norms = []
    for alpha in alphas:
        hook = alpha_hook(direction, alpha, mode="suppress")
        modified_output = hook(None, None, output)
        norms.append(torch.norm(output - modified_output).item())
    
    # Check that larger alpha causes larger modification
    assert norms[1] > norms[0]  # alpha=1.0 > alpha=0.1
    assert norms[2] > norms[1]  # alpha=10.0 > alpha=1.0


def test_alpha_hook_zero_direction():
    direction = torch.zeros(10)
    output = torch.randn(1, 5, 10)
    alpha = 1.0
    
    # Test with zero direction
    hook = alpha_hook(direction, alpha, mode="suppress")
    modified_output = hook(None, None, output)
    
    # Should be unchanged since direction is zero
    assert torch.allclose(output, modified_output, rtol=1e-5, atol=1e-5)


def test_alpha_hook_different_shapes():
    # Test with different batch sizes
    for batch_size in [1, 2, 4]:
        direction = torch.randn(10)
        output = torch.randn(batch_size, 5, 10)
        alpha = 1.0
        
        hook = alpha_hook(direction, alpha, mode="suppress")
        modified_output = hook(None, None, output)
        
        assert modified_output.shape == output.shape
        assert not torch.allclose(output, modified_output)
    
    # Test with different sequence lengths
    for seq_len in [1, 5, 10]:
        direction = torch.randn(10)
        output = torch.randn(1, seq_len, 10)
        alpha = 1.0
        
        hook = alpha_hook(direction, alpha, mode="suppress")
        modified_output = hook(None, None, output)
        
        assert modified_output.shape == output.shape
        assert not torch.allclose(output, modified_output) 