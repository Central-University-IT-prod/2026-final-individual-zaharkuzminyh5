"""
Скрипт обучения модели для DVC пайплайна.
"""

import json
import pickle
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.ml.model import LookalikeModel


def train_model(params_path: str = "params.yaml"):
    """
    Основная функция обучения модели.
    """
    print("=" * 50)
    print("Training model...")
    print("=" * 50)

    # Загрузка параметров
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
    model_params = params.get("model", {})
    print(f"Model params: {model_params}")

    # Пути к данным
    data_dir = Path("data")
    models_dir = Path("models")
    reports_dir = Path("reports")
    models_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)

    # Загрузка данных
    print("\nLoading data...")
    try:
        interactions = pd.read_parquet(data_dir / "interactions.parquet")
        features = pd.read_parquet(data_dir / "features.parquet")
        print(f"Loaded interactions: {interactions.shape}")
        print(f"Loaded features: {features.shape}")
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Please run the 'build_features' stage first.")
        return

    # Разделение на обучающий и тестовый наборы
    print("\nSplitting data into train and test sets...")
    train_interactions, test_interactions = train_test_split(
        interactions, test_size=0.2, random_state=42
    )
    print(f"Train interactions: {train_interactions.shape}")
    print(f"Test interactions: {test_interactions.shape}")

    # Сохранение тестового набора
    test_path = data_dir / "interactions_test.parquet"
    test_interactions.to_parquet(test_path, index=False)
    print(f"Test set saved to {test_path}")

    # Инициализация и обучение модели
    model = LookalikeModel(
        factors=model_params.get("n_factors", 128),
        iterations=model_params.get("n_iterations", 15),
        regularization=model_params.get("regularization", 0.01),
        random_state=42,
    )

    metrics = model.fit(train_interactions, features)

    # Сохранение модели
    model_path = models_dir / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved to {model_path}")

    # Сохранение метрик
    metrics_path = reports_dir / "train_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Train metrics saved to {metrics_path}")

    print("\n" + "=" * 50)
    print("Model training completed!")
    print("=" * 50)


if __name__ == "__main__":
    train_model()
