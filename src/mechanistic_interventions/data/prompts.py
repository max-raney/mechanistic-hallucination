import numpy as np
import re
from typing import List
from collections import defaultdict

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



