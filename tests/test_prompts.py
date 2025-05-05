import pytest
import numpy as np
import json
import pandas as pd
import yaml
from pathlib import Path
from mechanistic_interventions.data.prompts import (
    Vectorizer, 
    Regression, 
    PromptLoader,
    Prompt,
    PromptDifficulty
)


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


def test_prompt_loader():
    # Create temporary test files
    test_json = Path("test_prompts.json")
    test_csv = Path("test_prompts.csv")
    test_md = Path("test_prompts.md")
    test_yaml = Path("test_prompts.yaml")
    
    # Create test data
    test_data = [
        {"prompt": "test prompt 1", "category": "category1"},
        {"prompt": "test prompt 2", "category": "category2"}
    ]
    
    # Write test JSON file
    with open(test_json, 'w') as f:
        json.dump(test_data, f)
    
    # Write test CSV file
    df = pd.DataFrame(test_data)
    df.to_csv(test_csv, index=False)
    
    # Write test markdown file
    with open(test_md, 'w', encoding='utf-8') as f:
        f.write("""#### 1. Category1 Prompts
1. test prompt 1
2. test prompt 2

#### 2. Category2 Prompts
1. test prompt 3
2. test prompt 4
""")
    
    # Write test YAML file
    yaml_data = {
        "categories": {
            "category1": {
                "name": "Category1 Prompts",
                "description": "Test category 1",
                "prompts": [
                    {
                        "text": "test prompt 1",
                        "difficulty": "easy",
                        "tags": ["test"]
                    },
                    {
                        "text": "test prompt 2",
                        "difficulty": "medium",
                        "tags": ["test"]
                    }
                ]
            },
            "category2": {
                "name": "Category2 Prompts",
                "description": "Test category 2",
                "prompts": [
                    {
                        "text": "test prompt 3",
                        "difficulty": "hard",
                        "tags": ["test"]
                    }
                ]
            }
        }
    }
    with open(test_yaml, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_data, f)
    
    try:
        # Test JSON loading
        loader = PromptLoader()
        loader.load_from_json(test_json)
        assert len(loader.get_prompts()) == 2
        assert len(loader.get_categories()) == 2
        assert loader.get_prompts()[0] == "test prompt 1"
        assert loader.get_categories()[0] == "category1"
        assert len(loader.get_unique_categories()) == 2
        
        # Test CSV loading
        loader = PromptLoader()
        loader.load_from_csv(test_csv)
        assert len(loader.get_prompts()) == 2
        assert len(loader.get_categories()) == 2
        assert loader.get_prompts()[0] == "test prompt 1"
        assert loader.get_categories()[0] == "category1"
        assert len(loader.get_unique_categories()) == 2
        
        # Test markdown loading - all categories
        loader = PromptLoader()
        loader.load_from_markdown(test_md)
        assert len(loader.get_prompts()) == 4
        assert len(loader.get_categories()) == 4
        assert loader.get_prompts()[0] == "test prompt 1"
        assert loader.get_categories()[0] == "Category1 Prompts"
        assert len(loader.get_unique_categories()) == 2
        
        # Test markdown loading - specific category
        loader = PromptLoader()
        loader.load_from_markdown(test_md, category="Category1 Prompts")
        assert len(loader.get_prompts()) == 2
        assert len(loader.get_categories()) == 2
        assert loader.get_prompts()[0] == "test prompt 1"
        assert loader.get_categories()[0] == "Category1 Prompts"
        assert len(loader.get_unique_categories()) == 1
        
        # Test YAML loading - all categories
        loader = PromptLoader()
        loader.load_from_yaml(test_yaml)
        assert len(loader.get_prompts()) == 3
        assert len(loader.get_categories()) == 3
        assert loader.get_prompts()[0] == "test prompt 1"
        assert loader.get_categories()[0] == "Category1 Prompts"
        assert len(loader.get_unique_categories()) == 2
        
        # Test YAML loading - specific category
        loader = PromptLoader()
        loader.load_from_yaml(test_yaml, category="Category1 Prompts")
        assert len(loader.get_prompts()) == 2
        assert len(loader.get_categories()) == 2
        assert loader.get_prompts()[0] == "test prompt 1"
        assert loader.get_categories()[0] == "Category1 Prompts"
        assert len(loader.get_unique_categories()) == 1
        
        # Test difficulty filtering
        loader = PromptLoader()
        loader.load_from_yaml(test_yaml)
        easy_prompts = loader.get_prompts_by_difficulty(PromptDifficulty.EASY)
        assert len(easy_prompts) == 1
        assert easy_prompts[0].text == "test prompt 1"
        
        # Test tag filtering
        tagged_prompts = loader.get_prompts_by_tag("test")
        assert len(tagged_prompts) == 3
        
    finally:
        # Clean up test files
        test_json.unlink()
        test_csv.unlink()
        test_md.unlink()
        test_yaml.unlink() 