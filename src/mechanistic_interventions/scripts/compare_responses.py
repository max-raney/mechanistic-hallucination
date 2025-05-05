import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse
from pathlib import Path

def load_model_and_tokenizer(model_name, device):
    """Load the model and tokenizer."""
    print(f"Loading model from {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_new_tokens=1000, temperature=0.7):
    """Generate a response for a given prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def load_test_results(results_path):
    """Load the test results from the JSON file."""
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description='Compare original and intervention responses')
    parser.add_argument('--model', type=str, default="google/gemma-2b",
                      help='Model to use for generation')
    parser.add_argument('--results', type=str, default="test_results.json",
                      help='Path to test results JSON file')
    parser.add_argument('--output', type=str, default="comparison_results.json",
                      help='Path to save comparison results')
    parser.add_argument('--max_new_tokens', type=int, default=1000,
                      help='Maximum number of new tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7,
                      help='Temperature for generation')
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model, "cuda")
    
    # Load test results
    test_results = load_test_results(args.results)
    
    comparison_results = {}
    
    # Process each prompt
    for prompt, results in tqdm(test_results.items(), desc="Processing prompts"):
        comparison_results[prompt] = {
            "original": generate_response(model, tokenizer, prompt, args.max_new_tokens, args.temperature),
            "deception_suppress": generate_response(model, tokenizer, results["deception_suppress"], args.max_new_tokens, args.temperature),
            "deception_enhance": generate_response(model, tokenizer, results["deception_enhance"], args.max_new_tokens, args.temperature),
            "hallucination_suppress": generate_response(model, tokenizer, results["hallucination_suppress"], args.max_new_tokens, args.temperature),
            "hallucination_enhance": generate_response(model, tokenizer, results["hallucination_enhance"], args.max_new_tokens, args.temperature),
            "refusal_suppress": generate_response(model, tokenizer, results["refusal_suppress"], args.max_new_tokens, args.temperature),
            "refusal_enhance": generate_response(model, tokenizer, results["refusal_enhance"], args.max_new_tokens, args.temperature)
        }
    
    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(comparison_results, f, indent=2, ensure_ascii=False)
    
    print(f"Comparison results saved to {args.output}")

if __name__ == "__main__":
    main() 