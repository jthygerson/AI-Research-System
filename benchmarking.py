# benchmarking.py

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris  # Example dataset
import numpy as np

def run_benchmarks(model):
    """
    Runs benchmarking tests on the provided model using standard datasets.
    
    Parameters:
        model: The machine learning model developed by the AI Research System.
               The model should implement fit() and predict() methods.
               
    Returns:
        benchmarking_results: A dictionary containing evaluation metrics.
    """
    # Step 1: Load Benchmark Dataset
    # Replace load_iris() with relevant datasets for your research
    data = load_iris()
    X = data.data
    y = data.target
    dataset_name = 'Iris Dataset'

    # Step 2: Split Data into Training and Testing Sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Step 3: Train the Model
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"Error training model: {e}")
        return None

    # Step 4: Make Predictions
    try:
        y_pred = model.predict(X_test)
    except Exception as e:
        print(f"Error making predictions: {e}")
        return None

    # Step 5: Calculate Evaluation Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred)

    benchmarking_results = {
        "dataset": dataset_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": conf_matrix.tolist(),  # Convert to list for JSON serialization
    }

    return benchmarking_results
