import numpy as np
import torch
import argparse
import os

label_map = {
    "hallucination": 0,
    "deception": 1,
    "history": 2,
    "refusal": 3,
}

parser = argparse.ArgumentParser()
parser.add_argument("--class", dest="target_class", type=str, required=True, help="Target label name (e.g. hallucination)")
parser.add_argument("--layer", type=int, default=16)
parser.add_argument("--feat_path", type=str, default="gemma_prompt_activations.npy")
parser.add_argument("--label_path", type=str, default="data/prompt_labels.txt")
parser.add_argument("--save_path", type=str, default="concept_vectors")
args = parser.parse_args()

# Load activations and labels
X = np.load(args.feat_path)                           # [N, D]
with open(args.label_path, "r") as f:
    labels = [line.strip().lower() for line in f if line.strip()]
y = np.array([label_map.get(label, 0) for label in labels])   # fallback to 0

# Check length match
assert len(X) == len(y), "Number of activations and labels must match!"

# Compute direction
target_id = label_map[args.target_class]
X_pos = X[y == target_id]
X_neg = X[y != target_id]

direction = torch.tensor(X_pos.mean(0) - X_neg.mean(0)).float()
direction = direction / direction.norm()

# Save
os.makedirs(args.save_path, exist_ok=True)
save_name = f"{args.save_path}/{args.target_class}_mean_diff.pt"
torch.save(direction, save_name)
print(f"Mean-diff vector saved to: {save_name}")
