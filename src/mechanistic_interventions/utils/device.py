import torch
from contextlib import nullcontext

def get_default_device() -> str:
    """Return 'cuda' if available else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"

def autocast_context(device: str):
    """
    Return a mixed‐precision context manager:
    - If device=='cuda', import and use torch.cuda.amp.autocast()
    - Otherwise return a no-op nullcontext()
    """
    if device == "cuda":
        try:
            return torch.amp.autocast(device)
        except AttributeError:
            # fallback if torch.amp not present
            return nullcontext()
    else:
        return nullcontext()
    
def autocast(device: str):
    """
    Runtime‐dispatching autocast context manager.
    On 'cuda': returns torch.cuda.amp.autocast()
    On 'cpu':   returns a no-op nullcontext()
    """
    return autocast_context(device)
