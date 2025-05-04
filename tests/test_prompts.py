import pytest
import numpy as np
from mechanistic_interventions.data.prompts import Vectorizer, Regression


def test_vectorizer():
    vectorizer = Vectorizer()
    prompts = ["hello world", "world hello"]
    
    # Test fit
    vectorizer.fit(prompts)
    assert len(vectorizer.vocab) == 2
    assert "hello" in vectorizer.vocab
    assert "world" in vectorizer.vocab
    
    # Test transform
    X = vectorizer.transform(prompts)
    assert X.shape == (2, 2)
    assert np.all(X[0] == X[1])  # Same words, different order
    
    # Test fit_transform
    vectorizer = Vectorizer()
    X = vectorizer.fit_transform(prompts)
    assert X.shape == (2, 2)


def test_regression():
    # Create dummy data
    X = np.array([[1, 2], [2, 1], [3, 3]])
    y = ["class1", "class2", "class1"]
    
    # Test initialization
    reg = Regression(learning_rate=0.1, iter_times=100)
    assert reg.learning_rate == 0.1
    assert reg.iter_times == 100
    
    # Test fit
    reg.fit(X, y)
    assert reg.weights is not None
    assert reg.bias is not None
    assert len(reg.classes) == 2
    
    # Test predict
    predictions = reg.predict(X)
    assert len(predictions) == len(y)
    assert all(pred in ["class1", "class2"] for pred in predictions) 