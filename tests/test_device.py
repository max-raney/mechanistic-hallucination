import pytest
import torch
import contextlib
from mechanistic_interventions.utils.device import get_default_device, autocast, autocast_context


def test_get_default_device():
    device = get_default_device()
    assert device in ["cuda", "cpu"]
    if torch.cuda.is_available():
        assert device == "cuda"
    else:
        assert device == "cpu"


def test_autocast_context():
    # Test CPU path
    context = autocast_context("cpu")
    assert isinstance(context, type(contextlib.nullcontext()))
    
    # Test CUDA path if available
    if torch.cuda.is_available():
        context = autocast_context("cuda")
        assert isinstance(context, torch.amp.autocast)


def test_autocast():
    # Test CPU path
    context = autocast("cpu")
    assert isinstance(context, type(contextlib.nullcontext()))
    
    # Test CUDA path if available
    if torch.cuda.is_available():
        context = autocast("cuda")
        assert isinstance(context, torch.amp.autocast) 