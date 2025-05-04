import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# Import functions from our scripts
from src.mechanistic_interventions.scripts.inference_with_intervention import load_model_and_tokenizer
from src.mechanistic_interventions.utils.device import get_default_device
from src.mechanistic_interventions.data.prompts import PromptLoader

def main():
    # Set up device
    device = get_default_device()
    print(f"Using device: {device}")

    # Load a small model for testing (e.g., 'gpt2')
    model_path = "gpt2"
    model, tokenizer = load_model_and_tokenizer(model_path, device)
    print(f"Loaded model: {model_path}")

    # Load prompts using PromptLoader
    prompt_loader = PromptLoader()
    prompt_loader.load_from_json('src/mechanistic_interventions/data/training_prompts.json')
    prompts = [prompt.text for prompt in prompt_loader.prompts]
    labels = [prompt.category for prompt in prompt_loader.prompts]

    # Extract activations for each prompt
    activations = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            # Use the last hidden state as the activation
            activation = outputs.last_hidden_state[:, -1, :].squeeze(0).cpu().numpy()
            activations.append(activation)

    activations = np.stack(activations)  # shape: (num_prompts, hidden_dim)

    # Compute mean activations for each category
    unique_labels = list(set(labels))
    mean_activations = {}
    for label in unique_labels:
        mask = np.array(labels) == label
        mean_activations[label] = np.mean(activations[mask], axis=0)

    # Compute direction vectors (difference between means)
    direction_vectors = {}
    for label1 in unique_labels:
        for label2 in unique_labels:
            if label1 != label2:
                direction = mean_activations[label1] - mean_activations[label2]
                direction = direction / np.linalg.norm(direction)  # Normalize
                direction_vectors[f"{label1}_vs_{label2}"] = direction

    # Save direction vectors
    os.makedirs('direction_vectors', exist_ok=True)
    for name, direction in direction_vectors.items():
        np.save(f'direction_vectors/{name}.npy', direction)
        print(f"Saved direction vector for {name}")

if __name__ == "__main__":
    main() 