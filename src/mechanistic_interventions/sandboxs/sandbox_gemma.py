from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# ---- 基本配置 ----
tk = "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"
device = "cuda" if torch.cuda.is_available() else "cpu"
gemma_path = "/content/models/google_gemma-2b"
prompt_file = "data/clean_prompts.txt"  # 你的手动筛选后的prompt文件

# ---- 加载模型 ----
def load_gemma_model():
    print(f"Loading Gemma from {gemma_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        gemma_path,
        token=tk,
        torch_dtype=torch.float16 if 'cuda' in device else torch.float32
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(gemma_path, token=tk)
    print("Loaded Gemma model and tokenizer.")
    return model, tokenizer

# ---- 注册hook ----
def wrap_gemma_with_hooks(model, layer_idx_to_hook):
    hidden_states = []

    def hook_fn(module, input, output):
        hidden_states.append(output)

    try:
        layer_module = model.model.layers[layer_idx_to_hook]
    except AttributeError:
        raise ValueError("Layer not found, check model structure")

    hook = layer_module.register_forward_hook(hook_fn)
    return hidden_states, hook

# ---- 可视化每层激活强度 ----
def scan_all_layers(model, tokenizer, prompt, direction_vector=None):
    num_layers = model.config.num_hidden_layers
    scores = []

    for i in range(num_layers):
        activations, hook = wrap_gemma_with_hooks(model, i)
        _ = model(**tokenizer(prompt, return_tensors="pt").to(device))
        act_tensor = activations[0]
        if isinstance(act_tensor, tuple):
            act_tensor = act_tensor[0]
        act = act_tensor[:, -1, :].squeeze(0)
        score = (
            torch.dot(act, direction_vector.to(act.device)).item()
            if direction_vector is not None else
            torch.norm(act).item()
        )
        scores.append(score)
        hook.remove()

    return scores

# ---- 从文件加载 prompts ----
def load_prompts(path):
    with open(path, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts

# ---- 提取特征并保存 ----
def extract_features_for_prompts(model, tokenizer, prompts, layer_idx, save_path):
    features = []
    for prompt in prompts:
        activations, hook = wrap_gemma_with_hooks(model, layer_idx)
        _ = model(**tokenizer(prompt, return_tensors="pt").to(device))
        act_tensor = activations[0]
        if isinstance(act_tensor, tuple):
            act_tensor = act_tensor[0]
        vec = act_tensor[:, -1, :].squeeze(0).detach().cpu().numpy()
        features.append(vec)
        hook.remove()
    features = np.stack(features)
    np.save(save_path, features)
    print(f"Saved prompt features to {save_path}")

# ---- 主流程 ----
if __name__ == "__main__":
    model, tokenizer = load_gemma_model()
    
    # Step 1：可视化单个 prompt 的激活（帮助你选层）
    sample_prompt = "Why do humans lie?"
    scores = scan_all_layers(model, tokenizer, sample_prompt)
    best_layer = max(enumerate(scores), key=lambda x: x[1])[0]
    print(f"Most activated layer: {best_layer}")
    plt.plot(scores)
    plt.xlabel("Layer")
    plt.ylabel("Activation Strength")
    plt.title("Prompt Activation Across Layers")
    plt.grid(True)
    plt.show()

    # Step 2：批量处理多个 prompts 并提取激活
    prompts = load_prompts(prompt_file)
    save_path = "gemma_prompt_activations.npy"
    extract_features_for_prompts(model, tokenizer, prompts, best_layer, save_path)
