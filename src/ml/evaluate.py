"""
Скрипт оценки модели для DVC пайплайна.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from src.ml.model import LookalikeModel


def apk(actual, predicted, k=10):
    """
    Вычисляет Average Precision at k.
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """
    Вычисляет Mean Average Precision at k.
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def evaluate_model(params_path: str = "params.yaml"):
    """
    Основная функция оценки модели.
    """
    print("=" * 50)
    print("Evaluating model...")
    print("=" * 50)

    # Загрузка параметров
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
    model_params = params.get("model", {})
    k = model_params.get("top_n_default", 100)
    print(f"Evaluation params: k={k}")

    # Пути
    data_dir = Path("data")
    models_dir = Path("models")
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    # Загрузка модели
    print("\nLoading model...")
    model_path = models_dir / "model.pkl"
    try:
        with open(model_path, "rb") as f:
            model: LookalikeModel = pickle.load(f)
        print(f"Model loaded from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return

    # Загрузка тестовых данных
    print("\nLoading test data...")
    test_path = data_dir / "interactions_test.parquet"
    try:
        test_df = pd.read_parquet(test_path)
        print(f"Loaded test data: {test_df.shape}")
    except FileNotFoundError:
        print(f"Error: Test data not found at {test_path}")
        return

    # Подготовка ground truth
    actual_items = test_df.groupby("user_id").apply(
        lambda x: x.sort_values("confidence", ascending=False)["offer_id"].tolist()
    )
    test_users = actual_items.index.tolist()
    print(f"Found {len(test_users)} users in test set for evaluation.")

    # Получение предсказаний
    print(f"\nGenerating recommendations for {len(test_users)} users...")
    predicted_items = []
    for user_id in tqdm(test_users):
        recs, _ = model.recommend(user_id, top_n=k)
        predicted_items.append(list(recs))

    # Вычисление метрики
    map_at_k = mapk(actual_items.values, predicted_items, k=k)
    print(f"\nMAP@{k}: {map_at_k:.4f}")

    # Сохранение метрик
    eval_metrics = {"map_at_100": map_at_k}
    metrics_path = reports_dir / "eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(eval_metrics, f, indent=4)
    print(f"Evaluation metrics saved to {metrics_path}")

    # Сохранение отчета
    eval_report = {"map_at_100": map_at_k, "num_test_users": len(test_users), "k": k}
    report_path = reports_dir / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(eval_report, f, indent=4)
    print(f"Evaluation report saved to {report_path}")

    print("\n" + "=" * 50)
    print("Model evaluation completed!")
    print("=" * 50)


if __name__ == "__main__":
    evaluate_model()
