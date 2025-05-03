# scripts/run_prompts_and_save.py

import os
import torch
import argparse
from tqdm import tqdm

# sandbox
def load_sandbox(model_name):
    if model_name == "gemma":
        from src.mechanistic_interventions.sandboxs import sandbox_gemma as sandbox
    elif model_name == "llama":
        from src.mechanistic_interventions.sandboxs import sandbox_llama as sandbox
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return sandbox

# Loading prompts（txt or jsonl）
def load_prompts(prompt_path):
    if prompt_path.endswith(".txt"):
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
    elif prompt_path.endswith(".jsonl"):
        import json
        prompts = []
        with open(prompt_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                prompts.append(item.get("text") or item.get("prompt"))
    else:
        raise ValueError("Prompt file must be .txt or .jsonl")
    return prompts


def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    sandbox = load_sandbox(args.model)
    model, tokenizer = sandbox.load_gemma_model() if args.model == "gemma" else sandbox.load_llama_model()

    prompts = load_prompts(args.prompt_path)

    for i, prompt in enumerate(tqdm(prompts, desc="Running prompts")):
        scores = sandbox.scan_all_layers(model, tokenizer, prompt, direction_vector=None)

        save_path = os.path.join(args.save_dir, f"activation_{i:03d}.pt")
        torch.save({
            "prompt": prompt,
            "scores": scores  # List of float or torch.Tensor
        }, save_path)

    print(f"Saved {len(prompts)} activations to {args.save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["gemma", "llama"], required=True, help="Model to use")
    parser.add_argument("--prompt_path", type=str, required=True, help="Path to prompt file (.txt or .jsonl)")
    parser.add_argument("--save_dir", type=str, default="outputs", help="Directory to save .pt files")
    args = parser.parse_args()

    main(args)

