import time
import torch
import json
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.cuda.amp import autocast

# To introduce the models and test their performances
def benchmark_model(model_id, input_prompt="Hello, my name is"):  # For example
    print(f"\n--- Benchmarking {model_id} ---")

    # Load model
    t0 = time.time()
    tk = "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"  # Created by RY
    tokenizer = AutoTokenizer.from_pretrained(model_id,
                                             token = tk)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        token = tk)
    torch.cuda.synchronize()
    t1 = time.time()
    load_time = t1 - t0

    # Clear and track memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # Encode prompt
    inputs = tokenizer(input_prompt, return_tensors="pt").to("cuda")

    # Run generation
    with torch.no_grad(), autocast():
        t2 = time.time()
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False
        )
        torch.cuda.synchronize()
        t3 = time.time()

    inference_time = t3 - t2
    tokens_generated = outputs.shape[1] - inputs["input_ids"].shape[1]
    tokens_per_sec = tokens_generated / inference_time

    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)  # in GB
    decode = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {
        "model_id": model_id,
        "load_time_sec": load_time,
        "inference_time_sec": inference_time,
        "peak_memory_gb": peak_mem,
        "tokens_per_sec": tokens_per_sec,
        "output_text": decode
    }

# Run benchmark on multiple models
result1 = benchmark_model("google/gemma-2b")
result2 = benchmark_model("meta-llama/Meta-Llama-3-8B")

results = []
results.append(result1)
results.append(result2)

# Save results
with open("benchmark_results.json", "w") as f:
    json.dump(results, f, indent=2)
