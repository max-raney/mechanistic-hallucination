import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import joblib

def load_model_and_tokenizer(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def make_hook(concept_direction, alpha, target_class):
    def hook_fn(module, input, output):
        intervened_output = output.clone()
        direction = concept_direction[:, target_class]
        intervened_output[:, -1, :] += alpha * direction
        return intervened_output
    return hook_fn

def main(args):
    model, tokenizer = load_model_and_tokenizer(args.model_path)

    # Concept vector
    clf = joblib.load(args.concept_vector_path)
    concept_direction = torch.tensor(clf.coef_, dtype=torch.float32).T.cuda()  # shape: (hidden_dim, num_classes)

    # Hook
    target_layer = model.model.layers[args.layer]
    hook_handle = target_layer.mlp.register_forward_hook(
        make_hook(concept_direction, args.alpha, args.target_class)
    )

    inputs = tokenizer(args.prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n=== Intervention Output ===\n{decoded}")

    hook_handle.remove()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt to intervene on")
    parser.add_argument("--target_class", type=int, required=True, choices=[0, 1, 2, 3],
                        help="Concept class: 0=hallucination, 1=deception, 2=history, 3=refusal")
    parser.add_argument("--alpha", type=float, default=5.0, help="Scaling factor for concept direction")
    parser.add_argument("--layer", type=int, default=10, help="Layer to intervene on")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--model_path", type=str, default="/content/models/google_gemma-2b",
                        help="Path to Gemma model")
    parser.add_argument("--concept_vector_path", type=str, default="gemma_concept_vector_clf.pkl",
                        help="Path to trained concept vector classifier")

    args = parser.parse_args()
    main(args)
