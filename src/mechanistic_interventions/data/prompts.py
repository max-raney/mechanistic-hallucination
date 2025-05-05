import numpy as np
import re
import json
import yaml
import pandas as pd
from typing import List, Dict, Union, Optional
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

class PromptDifficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

@dataclass
class Prompt:
    text: str
    category: str
    difficulty: Optional[PromptDifficulty] = None
    tags: Optional[List[str]] = None

class PromptLoader:
    def __init__(self):
        self.prompts: List[Prompt] = []
        
    def load_from_json(self, file_path: Union[str, Path]) -> None:
        """Load prompts from a JSON file.
        
        Args:
            file_path: Path to the JSON file containing prompts and categories
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
            for item in data:
                self.prompts.append(Prompt(
                    text=item['prompt'],
                    category=item['category']
                ))
            
    def load_from_csv(self, file_path: Union[str, Path]) -> None:
        """Load prompts from a CSV file.
        
        Args:
            file_path: Path to the CSV file containing prompts and categories
        """
        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            self.prompts.append(Prompt(
                text=row['prompt'],
                category=row['category']
            ))
        
    def load_from_markdown(self, file_path: Union[str, Path], category: str = None) -> None:
        """Load prompts from a markdown file.
        
        Args:
            file_path: Path to the markdown file containing prompts
            category: Optional category to filter prompts by. If None, loads all prompts.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Split content into sections
        sections = re.split(r'#### \d+\.\s+', content)[1:]  # Skip the first empty split
        
        for section in sections:
            # Extract category name and prompts
            lines = section.strip().split('\n')
            section_category = lines[0].strip()
            
            if category and section_category != category:
                continue
                
            # Process each prompt line
            for line in lines[1:]:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                # Remove numbering if present
                prompt_text = re.sub(r'^\d+\.\s+', '', line)
                if prompt_text:
                    self.prompts.append(Prompt(
                        text=prompt_text,
                        category=section_category
                    ))

    def load_from_yaml(self, file_path: Union[str, Path], category: str = None) -> None:
        """Load prompts from a YAML file.
        
        Args:
            file_path: Path to the YAML file containing prompts
            category: Optional category to filter prompts by. If None, loads all prompts.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            
        # Get category name from the YAML data
        cat_name = data['name']
        if category and cat_name != category:
            return
            
        # Process prompts directly from the root level
        for prompt_data in data['prompts']:
            self.prompts.append(Prompt(
                text=prompt_data['text'],
                category=cat_name,
                difficulty=PromptDifficulty(prompt_data.get('difficulty', 'medium')),
                tags=prompt_data.get('tags', [])
            ))
        
    def get_prompts(self) -> List[str]:
        """Get the loaded prompts.
        
        Returns:
            List of prompt texts
        """
        return [p.text for p in self.prompts]
    
    def get_categories(self) -> List[str]:
        """Get the loaded categories.
        
        Returns:
            List of categories
        """
        return [p.category for p in self.prompts]
    
    def get_unique_categories(self) -> List[str]:
        """Get unique categories from the loaded data.
        
        Returns:
            List of unique categories
        """
        return list(sorted(set(p.category for p in self.prompts)))
    
    def get_prompts_by_difficulty(self, difficulty: PromptDifficulty) -> List[Prompt]:
        """Get prompts filtered by difficulty.
        
        Args:
            difficulty: The difficulty level to filter by
            
        Returns:
            List of prompts with the specified difficulty
        """
        return [p for p in self.prompts if p.difficulty == difficulty]
    
    def get_prompts_by_tag(self, tag: str) -> List[Prompt]:
        """Get prompts filtered by tag.
        
        Args:
            tag: The tag to filter by
            
        Returns:
            List of prompts with the specified tag
        """
        return [p for p in self.prompts if p.tags and tag in p.tags]

class Vectorizer:
    def __init__(self):
        self.vocab = {}
        
    def transfer2tokens(self, prompt: str):
        return re.findall(r'\b\w+\b', prompt.lower())

    def fit(self, prompts: List[str]):
        i = 0
        for prompt in prompts:
            for word in self.transfer2tokens(prompt):
                if word not in self.vocab:
                    self.vocab[word] = i
                    i += 1

    def transform(self, prompts: List[str]):
        X = np.zeros((len(prompts), len(self.vocab)))
        for i, prompt in enumerate(prompts):
            word_count = defaultdict(int)
            for word in self.transfer2tokens(prompt):
                if word in self.vocab:
                    word_count[word] += 1
            for word, count in word_count.items():
                X[i, self.vocab[word]] = count
        return X

    def fit_transform(self, prompts: List[str]):
        self.fit(prompts)
        return self.transform(prompts)


class Regression:
    def __init__(self, learning_rate=0.1, iter_times=1000):
        self.learning_rate = learning_rate
        self.iter_times = iter_times
        self.weights = None
        self.bias = None
        self.classes = []

    def prob(self, z):
        z_stable = z - np.max(z, axis=1, keepdims=True)
        exp_values = np.exp(z_stable)
        row_sums = np.sum(exp_values, axis=1, keepdims=True)
        
        return exp_values / row_sums
    
    def fit(self, X: np.ndarray, y: List[str]):
        self.classes = list(sorted(set(y)))
        y_encoded = np.array([self.classes.index(label) for label in y])
        n_samples, n_features = X.shape
        n_classes = len(self.classes)
        self.weights = np.zeros((n_classes, n_features))
        self.bias = np.zeros(n_classes)

        for step in range(self.iter_times):
            logits = X.dot(self.weights.T) + self.bias
            prob = self.prob(logits)
            true_labels = np.eye(n_classes)[y_encoded]

            grad_weights = (prob - true_labels).T.dot(X) / n_samples
            grad_bias = np.mean(prob - true_labels, axis=0)
            
            self.weights -= self.learning_rate * grad_weights
            self.bias -= self.learning_rate * grad_bias


    def predict(self, X):
        #Calculate score of sample in each category
        result = X.dot(self.weights.T) + self.bias
        
        # find highset score
        predictions = []
        for row in result:
            best_class_index = np.argmax(row)
            predictions.append(self.classes[best_class_index])
        
        return predictions



