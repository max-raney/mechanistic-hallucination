import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer

from ..data.prompts import PromptLoader
from ..utils.device import get_default_device

class DirectionVectorTrainer:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        save_dir: str = "direction_vectors",
        device: Optional[str] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.save_dir = Path(save_dir)
        self.device = device or get_default_device()
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_activations(
        self,
        prompts: List[str],
        layer_idx: int = -1  # -1 means last layer
    ) -> torch.Tensor:
        """Extract activations for a list of prompts at a specific layer."""
        activations = []
        
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                # Get activations from specified layer
                hidden_states = outputs.hidden_states[layer_idx]
                # Take the last token's activation
                activation = hidden_states[0, -1, :].cpu()
                activations.append(activation)
                
        return torch.stack(activations)
    
    def compute_direction(
        self,
        positive_prompts: List[str],
        negative_prompts: List[str],
        layer_idx: int = -1
    ) -> torch.Tensor:
        """Compute direction vector between positive and negative examples."""
        pos_activations = self.extract_activations(positive_prompts, layer_idx)
        neg_activations = self.extract_activations(negative_prompts, layer_idx)
        
        # Compute mean activations
        pos_mean = pos_activations.mean(dim=0)
        neg_mean = neg_activations.mean(dim=0)
        
        # Compute direction vector
        direction = pos_mean - neg_mean
        # Normalize
        direction = direction / direction.norm()
        
        return direction
    
    def train_and_save(
        self,
        concept_name: str,
        positive_prompts: List[str],
        negative_prompts: List[str],
        layer_idx: int = -1,
        metadata: Optional[Dict] = None
    ) -> Path:
        """Train and save a direction vector for a concept."""
        direction = self.compute_direction(positive_prompts, negative_prompts, layer_idx)
        
        # Prepare save path and metadata
        save_path = self.save_dir / f"{concept_name}_layer{layer_idx}.pt"
        save_dict = {
            "direction": direction,
            "metadata": {
                "model_name": self.model.config._name_or_path,
                "layer_idx": layer_idx,
                "num_positive": len(positive_prompts),
                "num_negative": len(negative_prompts),
                **(metadata or {})
            }
        }
        
        # Save
        torch.save(save_dict, save_path)
        return save_path

def train_concept_vectors_from_json(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    json_path: str,
    save_dir: str = "direction_vectors",
    layer_idx: int = -1
) -> List[Path]:
    """Convenience function to train vectors from a JSON file."""
    trainer = DirectionVectorTrainer(model, tokenizer, save_dir)
    prompt_loader = PromptLoader()
    prompt_loader.load_from_json(json_path)
    
    # Group prompts by category
    prompts_by_category = {}
    for prompt in prompt_loader.prompts:
        if prompt.category not in prompts_by_category:
            prompts_by_category[prompt.category] = []
        prompts_by_category[prompt.category].append(prompt.text)
    
    # Train vectors for each category vs others
    saved_paths = []
    categories = list(prompts_by_category.keys())
    for i, category in enumerate(categories):
        # Positive examples are from this category
        positive_prompts = prompts_by_category[category]
        # Negative examples are from all other categories
        negative_prompts = []
        for other_cat in categories:
            if other_cat != category:
                negative_prompts.extend(prompts_by_category[other_cat])
        
        save_path = trainer.train_and_save(
            f"{category}",
            positive_prompts,
            negative_prompts,
            layer_idx,
            metadata={"categories": categories}
        )
        saved_paths.append(save_path)
    
    return saved_paths 