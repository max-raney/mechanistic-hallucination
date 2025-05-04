import torch
from pathlib import Path
from typing import Dict, Optional, Union
from dataclasses import dataclass

@dataclass
class DirectionVector:
    direction: torch.Tensor
    metadata: Dict
    name: str
    layer_idx: int

class DirectionVectorStore:
    def __init__(self, base_dir: Union[str, Path] = "direction_vectors"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, DirectionVector] = {}
        
    def get(
        self,
        name: str,
        layer_idx: Optional[int] = None,
        device: Optional[str] = None
    ) -> DirectionVector:
        """Load a direction vector by name and optionally layer index."""
        # Try to find the exact file if layer_idx is specified
        if layer_idx is not None:
            file_path = self.base_dir / f"{name}_layer{layer_idx}.pt"
            cache_key = str(file_path)
            if cache_key in self._cache:
                vec = self._cache[cache_key]
                return DirectionVector(
                    vec.direction.to(device) if device else vec.direction,
                    vec.metadata,
                    vec.name,
                    vec.layer_idx
                )
                
            if file_path.exists():
                return self._load_vector(file_path, name, device)
        
        # Otherwise, find all matching files and take the latest
        matching_files = list(self.base_dir.glob(f"{name}_layer*.pt"))
        if not matching_files:
            raise ValueError(f"No direction vector found for concept '{name}'")
            
        latest_file = max(matching_files, key=lambda p: p.stat().st_mtime)
        return self._load_vector(latest_file, name, device)
    
    def _load_vector(
        self,
        file_path: Path,
        name: str,
        device: Optional[str] = None
    ) -> DirectionVector:
        """Load a direction vector from a file."""
        data = torch.load(file_path)
        direction = data["direction"]
        if device:
            direction = direction.to(device)
            
        vector = DirectionVector(
            direction=direction,
            metadata=data["metadata"],
            name=name,
            layer_idx=data["metadata"]["layer_idx"]
        )
        
        # Cache the vector
        self._cache[str(file_path)] = vector
        return vector
    
    def list_available(self) -> Dict[str, list]:
        """List all available direction vectors grouped by concept."""
        vectors = {}
        for file_path in self.base_dir.glob("*_layer*.pt"):
            # Parse name from filename (remove _layerX.pt)
            name = "_".join(file_path.stem.split("_")[:-1])
            if name not in vectors:
                vectors[name] = []
            vectors[name].append(file_path)
        return vectors 