import subprocess
import os
from pathlib import Path
import argparse
import json
from tqdm import tqdm
import sys

def run_command(command):
    """Run a command and return its output."""
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        stdout, stderr = process.communicate()
        return process.returncode, stdout, stderr
    except Exception as e:
        print(f"Error running command: {e}")
        return -1, "", str(e)

def main():
    parser = argparse.ArgumentParser(description='Generate direction vectors for multiple models')
    parser.add_argument('--models', type=str, nargs='+', required=True,
                      help='List of model names to process')
    parser.add_argument('--prompts', type=str, default="src/mechanistic_interventions/data/training_prompts.json",
                      help='Path to training prompts')
    parser.add_argument('--output_dir', type=str, default="direction_vectors",
                      help='Base directory for output')
    parser.add_argument('--layer', type=int, default=-1,
                      help='Layer to use for direction vectors')
    parser.add_argument('--alpha', type=float, default=0.5,
                      help='Alpha value for testing')
    parser.add_argument('--skip_training', action='store_true',
                      help='Skip training if vectors already exist')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each model
    for model in tqdm(args.models, desc="Processing models"):
        model_name = model.split('/')[-1]
        model_dir = os.path.join(args.output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Check if vectors already exist
        vectors_exist = all(
            os.path.exists(os.path.join(model_dir, f"{concept}_layer{args.layer}.pt"))
            for concept in ["deception", "hallucination", "refusal"]
        )
        
        if vectors_exist and args.skip_training:
            print(f"\nSkipping training for {model} as vectors already exist")
        else:
            # Train direction vectors
            train_cmd = f"python -m mechanistic_interventions.scripts.train_direction_vectors \
                --model {model} \
                --prompts {args.prompts} \
                --save_dir {model_dir} \
                --layer {args.layer}"
            
            print(f"\nTraining vectors for {model}...")
            code, out, err = run_command(train_cmd)
            if code != 0:
                print(f"Error training {model}:")
                print(err)
                continue
                
        # Analyze direction vectors
        analyze_cmd = f"python -m mechanistic_interventions.scripts.analyze_direction_vectors \
            --vectors_dir {model_dir} \
            --output_dir analysis/{model_name}"
            
        print(f"\nAnalyzing vectors for {model}...")
        code, out, err = run_command(analyze_cmd)
        if code != 0:
            print(f"Error analyzing {model}:")
            print(err)
            continue
            
        # Test direction vectors
        test_cmd = f"python -m mechanistic_interventions.scripts.test_direction_vectors \
            --model {model} \
            --prompts {args.prompts} \
            --vectors_dir {model_dir} \
            --layer {args.layer} \
            --alpha {args.alpha}"
            
        print(f"\nTesting vectors for {model}...")
        code, out, err = run_command(test_cmd)
        if code != 0:
            print(f"Error testing {model}:")
            print(err)
            continue
            
        print(f"\nCompleted processing {model}")
        
    print("\nAll models processed!")

if __name__ == "__main__":
    main() 