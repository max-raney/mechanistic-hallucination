import argparse
import torch
import json
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer

from mechanistic_interventions.direction_vectors.registry import DirectionVectorRegistry
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

def load_test_prompts(prompts_path: str) -> List[str]:
    """Load test prompts from file."""
    if prompts_path.endswith('.json'):
        with open(prompts_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return [item['prompt'] for item in data]
    elif prompts_path.endswith('.txt'):
        with open(prompts_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    else:
        raise ValueError(f"Unsupported prompt file format: {prompts_path}")

def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    do_sample: bool = False
) -> str:
    """Generate response from model for given prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=0.7 if do_sample else 1.0
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def test_direction_vectors(
    model_path: str,
    prompts_path: str,
    vectors_dir: str,
    layer_idx: int = -1,
    alpha: float = 1.0,
    device: Optional[str] = None,
    max_new_tokens: int = 100
) -> Dict:
    """Test direction vectors by comparing responses with and without interventions."""
    device = device or get_default_device()
    model, tokenizer = load_model_and_tokenizer(model_path, device)
    
    # Load test prompts
    prompts = load_test_prompts(prompts_path)
    print(f"Loaded {len(prompts)} test prompts")
    
    # Initialize registry
    registry = DirectionVectorRegistry(model, vectors_dir)
    
    # Get available vectors
    available_vectors = registry.store.list_available()
    print(f"Found {len(available_vectors)} direction vectors:")
    for name, files in available_vectors.items():
        print(f"  - {name}: {len(files)} files")
    
    results = {}
    
    # Test each prompt
    for prompt in tqdm(prompts, desc="Testing prompts"):
        print(f"\nTesting prompt: {prompt[:50]}...")
        prompt_results = {"original": generate_response(model, tokenizer, prompt, max_new_tokens)}
        
        # Test each direction vector
        for concept_name, vector_files in available_vectors.items():
            print(f"  Testing concept: {concept_name}")
            # Try both suppress and enhance modes
            for mode in ["suppress", "enhance"]:
                print(f"    Mode: {mode}")
                # Apply intervention
                handle_id = registry.apply_intervention(
                    concept_name=concept_name,
                    layer_idx=layer_idx,
                    alpha=alpha,
                    mode=mode
                )
                
                # Generate response with intervention
                response = generate_response(model, tokenizer, prompt, max_new_tokens)
                
                # Store result
                key = f"{concept_name}_{mode}"
                prompt_results[key] = response
                
                # Remove intervention
                registry.remove_intervention(handle_id)
        
        results[prompt] = prompt_results
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Test direction vectors on a model")
    parser.add_argument("--model", type=str, required=True, help="Path to model or HuggingFace model ID")
    parser.add_argument("--prompts", type=str, required=True, help="Path to test prompts file (JSON or TXT)")
    parser.add_argument("--vectors_dir", type=str, required=True, help="Directory containing trained direction vectors")
    parser.add_argument("--layer", type=int, default=-1, help="Layer index to apply interventions (-1 for last layer)")
    parser.add_argument("--alpha", type=float, default=1.0, help="Alpha scale factor for interventions (default: 1.0)")
    parser.add_argument("--device", type=str, help="Device to use (cuda/cpu)")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum number of tokens to generate")
    parser.add_argument("--output", type=str, default="test_results.json", help="Path to save results")
    args = parser.parse_args()
    
    print(f"\nTesting direction vectors with:")
    print(f"  Model: {args.model}")
    print(f"  Layer: {args.layer}")
    print(f"  Alpha: {args.alpha}")
    print(f"  Device: {args.device or get_default_device()}")
    
    # Run tests
    results = test_direction_vectors(
        model_path=args.model,
        prompts_path=args.prompts,
        vectors_dir=args.vectors_dir,
        layer_idx=args.layer,
        alpha=args.alpha,
        device=args.device,
        max_new_tokens=args.max_new_tokens
    )
    
    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nTest results saved to {args.output}")

if __name__ == "__main__":
    main() 