import pytest
import os
import json
import pandas as pd
from mechanistic_interventions.data.train_and_predict import (
    load_training_data_from_csv,
    load_training_data_from_json,
    get_training_data,
    get_test_data
)


@pytest.fixture
def temp_csv(tmp_path):
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({
        'prompt': ['test prompt 1', 'test prompt 2'],
        'category': ['category1', 'category2']
    })
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def temp_json(tmp_path):
    json_path = tmp_path / "test.json"
    data = [
        {"prompt": "test prompt 1", "category": "category1"},
        {"prompt": "test prompt 2", "category": "category2"}
    ]
    with open(json_path, 'w') as f:
        json.dump(data, f)
    return json_path


def test_load_training_data_from_csv(temp_csv):
    prompts, categories = load_training_data_from_csv(temp_csv)
    assert len(prompts) == 2
    assert len(categories) == 2
    assert prompts[0] == 'test prompt 1'
    assert categories[0] == 'category1'


def test_load_training_data_from_json(temp_json):
    prompts, categories = load_training_data_from_json(temp_json)
    assert len(prompts) == 2
    assert len(categories) == 2
    assert prompts[0] == 'test prompt 1'
    assert categories[0] == 'category1'


def test_get_training_data(tmp_path, temp_csv, temp_json):
    # Test with both CSV and JSON
    prompts, categories = get_training_data()
    assert len(prompts) > 0
    assert len(categories) > 0
    assert len(prompts) == len(categories)
    
    # Test with only CSV
    os.remove(temp_json)
    prompts, categories = get_training_data()
    assert len(prompts) > 0
    assert len(categories) > 0
    
    # Test with only JSON
    os.remove(temp_csv)
    prompts, categories = get_training_data()
    assert len(prompts) > 0
    assert len(categories) > 0


def test_get_test_data(tmp_path, temp_csv, temp_json):
    # Test with both CSV and JSON
    prompts = get_test_data()
    assert len(prompts) > 0
    
    # Test with only CSV
    os.remove(temp_json)
    prompts = get_test_data()
    assert len(prompts) > 0
    
    # Test with only JSON
    os.remove(temp_csv)
    prompts = get_test_data()
    assert len(prompts) > 0 