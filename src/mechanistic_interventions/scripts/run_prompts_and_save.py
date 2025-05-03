# scripts/run_prompts_and_save.py

import os
import torch
import pickle
from tqdm import tqdm
from mechanistic_interventions.models.wrapper import load_model_wrapper

PROMPT_PATH = "prompt.md" 
MODEL_NAMES = ["google/gemma-2b", "meta-llama/Meta-Llama-3-8B"]
OUTPUT_DIR = "outputs"
LAYER_NAMES = ["resid"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Loading prompts for each line
with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    prompts = [line.strip() for line in f if line.strip()]

# Iterate each model
for model_name in MODEL_NAMES:
    print(f"Running model: {model_name}")
    wrapper = load_model_wrapper(model_name)  # mechanistic_interventions/models/xx_wrapper.py

    all_results = []

    for prompt in tqdm(prompts, desc=f"Processing {model_name}"):
        result = {
            "prompt": prompt,
            "activations": {},
            "output_text": None,
        }

        # Extract, output, and storage activations
        with wrapper.record_activations(LAYER_NAMES) as recorder:
            output_text = wrapper(prompt)

        result["output_text"] = output_text

        for layer in LAYER_NAMES:
            result["activations"][layer] = [
                t.clone().cpu() for t in recorder[layer]
            ]

        all_results.append(result)

    # Save as pt files
    short_name = model_name.split("/")[-1].replace("-", "_")
    save_path = os.path.join(OUTPUT_DIR, f"{short_name}_activations.pt")
    torch.save(all_results, save_path)
    print(f"Saved to {save_path}")
