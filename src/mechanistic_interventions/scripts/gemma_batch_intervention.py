import os
import json
import torch
import numpy as np
import joblib
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==================== CONFIG ====================
MODEL_PATH = "/content/models/google_gemma-2b"
PROMPT_FILE = "data/clean_prompt.txt"
LABEL_FILE = "data/prompt_labels.txt"
PKL_PATH = "gemma_concept_vector_clf.pkl"
OUTPUT_JSON = "results/gemma_intervention_outputs.json"

ALPHAS = [5, 10, 20]
MODES = ["suppress", "enhance"]
USE_AUTO_LAYER = True  # Auto find most activated layer for each prompt
MAX_NEW_TOKENS = 100

# ==================== SETUP =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# 4×hidden_dim matrix
clf = joblib.load(PKL_PATH)
directions = torch.tensor(clf.coef_, dtype=torch.float32).to(device)  # shape [4, D]
label2id = {"hallucination": 0, "deception": 1, "history": 2, "refusal": 3}


# =============== HELPERS ================

def load_prompts_labels(p_path, l_path):
    with open(p_path, "r", encoding="utf-8") as f:
        prompts = [ln.strip() for ln in f if ln.strip()]
    with open(l_path, "r", encoding="utf-8") as f:
        labels = [ln.strip().lower() for ln in f if ln.strip()]
    assert len(prompts) == len(labels), "Prompt & label mismatch"
    return prompts, labels


def alpha_hook(vec, alpha, mode):
    def hook(_, __, out):
        v = vec.to(out.dtype).to(out.device)         
        if mode == "suppress":
            proj = torch.einsum("bsd,d->bs", out, v)
            return out - alpha * proj.unsqueeze(-1) * v
        else:
            return out + alpha * v
    return hook


def best_layer(prompt):
    scores = []
    for i in range(model.config.num_hidden_layers):
        captured = []
        h = model.model.layers[i].register_forward_hook(lambda _, __, o: captured.append(o))
        _ = model(**tokenizer(prompt, return_tensors="pt").to(device))
        h.remove()
        t = captured[0][0] if isinstance(captured[0], tuple) else captured[0]
        scores.append(torch.norm(t[:, -1, :]).item())
    return int(np.argmax(scores))


def generate(prompt, layer, direction=None, alpha=None, mode="suppress"):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    handle = None
    if direction is not None and alpha is not None:
        handle = model.model.layers[layer].mlp.register_forward_hook(alpha_hook(direction, alpha, mode))
    out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    if handle:
        handle.remove()
    return tokenizer.decode(out[0], skip_special_tokens=True)


# ==================== MAIN =====================

def main():
    prompts, labels = load_prompts_labels(PROMPT_FILE, LABEL_FILE)
    os.makedirs("results", exist_ok=True)
    results = []

    for prompt, label in tqdm(list(zip(prompts, labels))):
        cls_id = label2id[label]
        vec = directions[cls_id]
        lyr = best_layer(prompt) if USE_AUTO_LAYER else 16

        entry = {
            "prompt": prompt,
            "label": label,
            "best_layer": lyr,
            "original": generate(prompt, lyr)
        }
        for mode in MODES:
            for alpha in ALPHAS:
                k = f"{mode}_α{alpha}"
                entry[k] = generate(prompt, lyr, vec, alpha, mode)
        results.append(entry)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved → {OUTPUT_JSON}  (total {len(results)} prompts)")


if __name__ == "__main__":
    main()
