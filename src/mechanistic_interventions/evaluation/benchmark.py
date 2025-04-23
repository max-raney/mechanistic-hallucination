# mechanistic_interventions/evaluation/benchmark.py
import time
import torch
from typing import Any, Dict, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.cuda.amp import autocast

# Token
from mechanistic_interventions.config import HF_TOKEN


def benchmark_model(
    model_id: str,
    prompt: str,
    token: Optional[str] = None,
    device: str = "cuda",
    max_new_tokens: int = 50,
    do_sample: bool = False,
) -> Dict[str, Any]:
    """
    Load a causal LM, run a generation, and return timing and memory metrics.

    Args:
        model_id: HuggingFace repo ID of the model.
        prompt:   Text prompt for generation.
        token:    HF authentication token (falls back to config.HF_TOKEN).
        device:   PyTorch device ("cuda" or "cpu").
        max_new_tokens: number of tokens to generate.
        do_sample: whether to sample (True) or greedy (False).

    Returns:
        A dict with load_time_sec, inference_time_sec, peak_memory_gb,
        tokens_per_sec, and the generated output text.
    """
    use_token = token or HF_TOKEN

    # 1) Measure model load time
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, use_auth_token=use_token
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto" if device == "cuda" else None,
        use_auth_token=use_token,
    )
    if device == "cuda":
        torch.cuda.synchronize()
    load_time = time.time() - t0

    # 2) Reset GPU memory stats
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # 3) Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # 4) Run generation and measure inference time
    with torch.no_grad(), autocast():
        t1 = time.time()
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        inference_time = time.time() - t1

    # 5) Compute metrics
    tokens_generated = outputs.shape[1] - inputs["input_ids"].shape[1]
    tokens_per_sec = tokens_generated / inference_time if inference_time > 0 else 0.0
    peak_memory = (
        torch.cuda.max_memory_allocated() / (1024 ** 3)
        if device == "cuda"
        else 0.0
    )
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 6) Return structured results
    return {
        "model_id": model_id,
        "load_time_sec": load_time,
        "inference_time_sec": inference_time,
        "peak_memory_gb": peak_memory,
        "tokens_per_sec": tokens_per_sec,
        "output_text": output_text,
    }

def main():
    # Hard-coded model list & prompt for now
    models = ["google/gemma-2b", "meta-llama/Meta-Llama-3-8B"]
    prompt = "Hello, my name is"
    from mechanistic_interventions.config import HF_TOKEN

    results = []
    for m in models:
        res = benchmark_model(m, prompt, token=HF_TOKEN)
        results.append(res)

    import json
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
