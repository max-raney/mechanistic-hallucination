from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import matplotlib.pyplot as plt


tk = "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"  # Well, I have to repeatly define this to avoid crash...
device = "cuda" if torch.cuda.is_available() else "cpu"
gemma_path = "/content/models/google_gemma-2b"


# Step1: Load Gemma-2b model
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


# Step2: Wrap Gemma model to capture its hidden states on all layers
def wrap_gemma_with_hooks(model, layer_idx_to_hook):
    hidden_states = []

    def hook_fn(module, input, output):
        hidden_states.append(output)
        # Also ok to output only the last layer's token：output[:, -1, :]，depend on particular task

    # General way to find transformer block, may vary to model type：
    # Name of layer may differ with model，e.g: model.layers[layer_idx]
    try:
        layer_module = model.model.layers[layer_idx_to_hook]
    except AttributeError:
        raise ValueError("Haven't detact layer，plz maually orient transformer layer")

    hook = layer_module.register_forward_hook(hook_fn)

    return hidden_states, hook


# Step 3: Analyze activation strength across layers, to find that particular one
def scan_all_layers(model, tokenizer, prompt, direction_vector=None):
    num_layers = model.config.num_hidden_layers
    scores = []

    for i in range(num_layers):
        activations, hook = wrap_gemma_with_hooks(model, i)
        _ = model(**tokenizer(prompt, return_tensors="pt").to(device))

        # Use the hidden state of the last token
        act_tensor = activations[0]
        if isinstance(act_tensor, tuple):  # Some model returns (hidden_state, ...)
            act_tensor = act_tensor[0]
        act = act_tensor[:, -1, :].squeeze(0)  # shape: [hidden_dim]

        if direction_vector is not None:
            # Using dot product to evaluate the degree of activation of each semantic direction
            score = torch.dot(act, direction_vector.to(act.device)).item()
        else:
            # Using vector-norm to illustrate the intense
            score = torch.norm(act).item()

        scores.append(score)
        hook.remove()

    return scores


# Step 4: Run everything and visualizing the results
gemma_model, gemma_tokenizer = load_gemma_model()
prompt = "Why do humans lie?"  # Example

dangerous_intent_dir = None  # Initialize to "None", or torch.randn(model.config.hidden_size)
scores = scan_all_layers(gemma_model, gemma_tokenizer, prompt, direction_vector=dangerous_intent_dir)
best_layer = max(enumerate(scores), key=lambda x: x[1])[0]
best_score = scores[best_layer]
print(f"Most activated layer: {best_layer} with score {best_score:.4f}")

plt.plot(scores)
plt.xlabel("Layer")
plt.ylabel("Activation Strength" if dangerous_intent_dir is None else "Dot with DangerousIntentDir")
plt.title("Prompt Activation Across Transformer Layers of the Gemma-2b model")
plt.grid(True)
plt.show()
