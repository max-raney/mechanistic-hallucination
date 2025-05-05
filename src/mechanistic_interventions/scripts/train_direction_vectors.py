import argparse
import torch
import os
from pathlib import Path
from typing import List, Dict, Optional
import yaml

from transformers import AutoModelForCausalLM, AutoTokenizer

from mechanistic_interventions.direction_vectors.train import DirectionVectorTrainer
from mechanistic_interventions.data.prompts import PromptLoader, Prompt
from mechanistic_interventions.utils.device import get_default_device
from mechanistic_interventions.config import HF_TOKEN

def load_model_and_tokenizer(model_path: str, device: str) -> tuple:
    """Load model and tokenizer from path."""
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        token=HF_TOKEN,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device).eval()
    return model, tokenizer

def train_vectors_for_model(
    model_path: str,
    prompts_path: str,
    save_dir: str,
    layer_idx: int = -1,
    device: Optional[str] = None
) -> List[Path]:
    """Train direction vectors for a specific model using prompts from a file."""
    device = device or get_default_device()
    model, tokenizer = load_model_and_tokenizer(model_path, device)
    
    # Load prompts
    prompt_loader = PromptLoader()
    if prompts_path.endswith('.json'):
        prompt_loader.load_from_json(prompts_path)
    elif prompts_path.endswith('.csv'):
        prompt_loader.load_from_csv(prompts_path)
    elif prompts_path.endswith('.yaml'):
        with open(prompts_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            for category_name, category_data in data['categories'].items():
                for prompt_data in category_data['prompts']:
                    prompt_loader.prompts.append(Prompt(
                        text=prompt_data['text'],
                        category=category_name,
                        difficulty=prompt_data.get('difficulty', 'medium'),
                        tags=prompt_data.get('tags', [])
                    ))
    elif prompts_path.endswith('.txt'):
        # For .txt files, assume each line is a prompt and assign a default category
        with open(prompts_path, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
            for prompt in prompts:
                prompt_loader.prompts.append(Prompt(
                    text=prompt,
                    category="default"  # Assign a default category
                ))
    else:
        raise ValueError(f"Unsupported prompt file format: {prompts_path}")
    
    # Initialize trainer
    trainer = DirectionVectorTrainer(model, tokenizer, save_dir, device)
    
    # Get unique categories
    categories = prompt_loader.get_unique_categories()
    saved_paths = []
    
    # Train vectors for each category
    for category in categories:
        # Get positive and negative examples
        positive_prompts = [p.text for p in prompt_loader.prompts if p.category == category]
        negative_prompts = [p.text for p in prompt_loader.prompts if p.category != category]
        
        if not positive_prompts or not negative_prompts:
            print(f"Skipping category {category} - insufficient examples")
            continue
            
        print(f"Training vector for category: {category}")
        save_path = trainer.train_and_save(
            concept_name=category,
            positive_prompts=positive_prompts,
            negative_prompts=negative_prompts,
            layer_idx=layer_idx,
            metadata={
                "model_name": model_path,
                "num_positive": len(positive_prompts),
                "num_negative": len(negative_prompts)
            }
        )
        saved_paths.append(save_path)
        print(f"Saved vector to: {save_path}")
    
    return saved_paths

def main():
    parser = argparse.ArgumentParser(description="Train direction vectors for language models")
    parser.add_argument("--model", type=str, required=True, help="Path to model or HuggingFace model ID")
    parser.add_argument("--prompts", type=str, required=True, help="Path to prompts file (JSON, CSV, or YAML)")
    parser.add_argument("--save_dir", type=str, default="direction_vectors", help="Directory to save vectors")
    parser.add_argument("--layer", type=int, default=-1, help="Layer index to extract activations from (-1 for last layer)")
    parser.add_argument("--device", type=str, help="Device to use (cuda/cpu)")
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Train vectors
    saved_paths = train_vectors_for_model(
        model_path=args.model,
        prompts_path=args.prompts,
        save_dir=args.save_dir,
        layer_idx=args.layer,
        device=args.device
    )
    
    print(f"\nTraining complete! Saved {len(saved_paths)} direction vectors to {args.save_dir}")

if __name__ == "__main__":
    main() 