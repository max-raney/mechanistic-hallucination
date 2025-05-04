import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# Update imports to use the package's modules
from mechanistic_interventions.utils.device import get_default_device


def load_model_and_tokenizer(model_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    model = model.to(device).eval()
    return model, tokenizer


def alpha_hook(direction, alpha, mode="suppress"):
    # Only normalize if direction is not zero
    if direction.norm() > 0:
        direction = direction / direction.norm()
    else:
        # For zero direction, return identity function
        return lambda module, input, output: output

    def hook_fn(module, input, output):
        if mode == "suppress":
            proj = torch.einsum("bsd,d->bs", output, direction)
            output = output - alpha * proj.unsqueeze(-1) * direction
        else:  # enhance mode
            proj = torch.einsum("bsd,d->bs", output, direction)
            output = output + alpha * proj.unsqueeze(-1) * direction
        return output

    return hook_fn


def main(args):
    device = get_default_device()
    model, tokenizer = load_model_and_tokenizer(args.model_path, device)

    # Load concept direction
    direction = torch.load(args.target_direction, map_location=device).float().to(device)

    # Register hook at specified layer
    target_layer = model.model.layers[args.layer]
    hook_handle = target_layer.mlp.register_forward_hook(
        alpha_hook(direction, alpha=args.alpha, mode=args.mode)
    )

    # Run inference
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n=== Intervention Output ({args.mode}, α={args.alpha}) ===\n{decoded}")

    hook_handle.remove()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt")
    parser.add_argument("--target_direction", type=str, required=True, help="Path to .pt file of direction vector")
    parser.add_argument("--alpha", type=float, default=10.0, help="Alpha scale factor")
    parser.add_argument("--layer", type=int, default=16, help="Layer index to intervene")
    parser.add_argument("--mode", type=str, default="suppress", choices=["suppress", "enhance"],
                        help="Intervention mode")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--model_path", type=str, default="/content/models/google_gemma-2b")

    args = parser.parse_args()
    main(args) 