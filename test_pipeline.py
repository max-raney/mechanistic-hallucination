import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# Import functions from our scripts
from src.mechanistic_interventions.scripts.inference_with_intervention import load_model_and_tokenizer
from src.mechanistic_interventions.utils.device import get_default_device

def alpha_hook(direction, alpha, mode="suppress"):
    # Only normalize if direction is not zero
    if direction.norm() > 0:
        direction = direction / direction.norm()
    else:
        # For zero direction, return identity function
        return lambda module, input, output: output

    def hook_fn(module, input, output):
        # GPT-2 layers return tuples, get the hidden states
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        # Convert direction to match hidden states dtype
        direction_matched = direction.to(dtype=hidden_states.dtype)

        if mode == "suppress":
            proj = torch.einsum("bsd,d->bs", hidden_states, direction_matched)
            hidden_states = hidden_states - alpha * proj.unsqueeze(-1) * direction_matched
        else:  # enhance mode
            proj = torch.einsum("bsd,d->bs", hidden_states, direction_matched)
            hidden_states = hidden_states + alpha * proj.unsqueeze(-1) * direction_matched
        
        # Return in the same format as received
        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states

    return hook_fn

def main():
    # Set up device
    device = get_default_device()
    print(f"Using device: {device}")

    # Load a small model for testing (e.g., 'gpt2')
    model_path = "gpt2"
    model, tokenizer = load_model_and_tokenizer(model_path, device)
    print(f"Loaded model: {model_path}")

    # Define test prompts
    prompts = [
        "The sky is",
        "Once upon a time",
        "In a galaxy far, far away"
    ]

    # Define a simple direction vector (for demonstration)
    # In a real scenario, this would be a trained concept vector
    direction = torch.randn(model.config.hidden_size, dtype=torch.float32).to(device)
    direction = direction / direction.norm()  # Normalize

    # Run interventions and compare outputs
    for prompt in prompts:
        print(f"\n=== Original Prompt: {prompt} ===")
        
        # Original output
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
            original_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Original Output: {original_output}")

            # Intervention output (suppress)
            target_layer = model.transformer.h[0]  # Using the first layer for simplicity
            hook_handle = target_layer.register_forward_hook(
                alpha_hook(direction, alpha=10.0, mode="suppress")
            )
            
            outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
            intervention_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Intervention Output (suppress): {intervention_output}")
            
            hook_handle.remove()

if __name__ == "__main__":
    main() 