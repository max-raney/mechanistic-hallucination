from typing import Dict, Optional, List, Callable
import torch
from torch.nn import Module
from pathlib import Path

from .store import DirectionVector, DirectionVectorStore
from ..utils.device import get_default_device

class DirectionVectorRegistry:
    def __init__(self, model: Module, base_dir: Optional[str] = None):
        self.model = model
        self.store = DirectionVectorStore(base_dir) if base_dir else DirectionVectorStore()
        self.active_hooks: Dict[str, List[torch.utils.hooks.RemovableHandle]] = {}
        
    def create_hook(
        self,
        direction: DirectionVector,
        alpha: float = 1.0,
        mode: str = "suppress"
    ) -> Callable:
        """Create a hook function for the given direction vector."""
        if direction.direction.norm() == 0:
            return lambda module, input, output: output
            
        def hook_fn(module, input, output):
            # Handle tuple outputs (common in transformer layers)
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
                
            # Convert direction to match hidden states dtype and device
            direction_matched = direction.direction.to(
                dtype=hidden_states.dtype,
                device=hidden_states.device
            )
            
            # Compute projection
            proj = torch.einsum("bsd,d->bs", hidden_states, direction_matched)
            
            # Apply intervention
            if mode == "suppress":
                hidden_states = hidden_states - alpha * proj.unsqueeze(-1) * direction_matched
            else:  # enhance mode
                hidden_states = hidden_states + alpha * proj.unsqueeze(-1) * direction_matched
                
            # Return in the same format as received
            if isinstance(output, tuple):
                return (hidden_states,) + output[1:]
            return hidden_states
            
        return hook_fn
        
    def apply_intervention(
        self,
        concept_name: str,
        layer_idx: Optional[int] = None,
        alpha: float = 1.0,
        mode: str = "suppress"
    ) -> str:
        """Apply an intervention for a concept and return a handle ID."""
        # Load direction vector
        direction = self.store.get(
            concept_name,
            layer_idx=layer_idx,
            device=get_default_device()
        )
        
        # Create hook
        hook_fn = self.create_hook(direction, alpha, mode)
        
        # Register hook
        if hasattr(self.model, "transformer"):
            # For transformer models
            if layer_idx is None:
                layer_idx = direction.layer_idx
            target_layer = self.model.transformer.h[layer_idx]
        elif hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            # For Gemma and similar architectures
            if layer_idx is None:
                layer_idx = direction.layer_idx
            target_layer = self.model.model.layers[layer_idx]
        else:
            # For other models - adjust as needed
            raise ValueError("Unsupported model architecture")
            
        hook_handle = target_layer.register_forward_hook(hook_fn)
        
        # Generate unique handle ID
        handle_id = f"{concept_name}_{layer_idx}_{mode}_{len(self.active_hooks)}"
        self.active_hooks[handle_id] = [hook_handle]
        
        return handle_id
        
    def remove_intervention(self, handle_id: str):
        """Remove an intervention by its handle ID."""
        if handle_id in self.active_hooks:
            for hook in self.active_hooks[handle_id]:
                hook.remove()
            del self.active_hooks[handle_id]
            
    def remove_all_interventions(self):
        """Remove all active interventions."""
        for handle_id in list(self.active_hooks.keys()):
            self.remove_intervention(handle_id)
            
    def list_active_interventions(self) -> Dict[str, Dict]:
        """List all active interventions."""
        return {
            handle_id: {
                "num_hooks": len(hooks),
                "concept": handle_id.split("_")[0],
                "layer": int(handle_id.split("_")[1]),
                "mode": handle_id.split("_")[2]
            }
            for handle_id, hooks in self.active_hooks.items()
        } 