import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional
import argparse
import json

from mechanistic_interventions.direction_vectors.store import DirectionVectorStore

def analyze_vector(vector: torch.Tensor) -> Dict:
    """Analyze a direction vector and return statistics."""
    stats = {
        "norm": vector.norm().item(),
        "mean": vector.mean().item(),
        "std": vector.std().item(),
        "min": vector.min().item(),
        "max": vector.max().item(),
        "num_zeros": (vector.abs() < 1e-6).sum().item(),
        "num_large": (vector.abs() > 0.1).sum().item()
    }
    return stats

def plot_vector_distribution(vector: torch.Tensor, title: str, save_path: Optional[Path] = None):
    """Plot the distribution of values in a direction vector."""
    plt.figure(figsize=(10, 6))
    plt.hist(vector.cpu().numpy(), bins=100)
    plt.title(f"Distribution of {title}")
    plt.xlabel("Value")
    plt.ylabel("Count")
    if save_path:
        plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze direction vectors")
    parser.add_argument("--vectors_dir", type=str, required=True, help="Directory containing direction vectors")
    parser.add_argument("--output_dir", type=str, default="analysis", help="Directory to save analysis results")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize store
    store = DirectionVectorStore(args.vectors_dir)
    
    # Get all vectors
    vectors = store.list_available()
    
    # Analyze each vector
    for concept_name, vector_files in vectors.items():
        print(f"\nAnalyzing concept: {concept_name}")
        
        # Load vector
        vector = store.get(concept_name)
        
        # Analyze vector
        stats = analyze_vector(vector.direction)
        print("Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value:.4f}")
        
        # Plot distribution
        plot_path = output_dir / f"{concept_name}_distribution.png"
        plot_vector_distribution(vector.direction, concept_name, plot_path)
        print(f"  Plot saved to: {plot_path}")
        
        # Save statistics
        stats_path = output_dir / f"{concept_name}_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"  Statistics saved to: {stats_path}")

if __name__ == "__main__":
    main() 