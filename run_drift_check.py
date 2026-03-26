"""
Скрипт для запуска проверки дрейфа данных.
"""

import json

# Добавляем путь к модулям
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src")

from monitoring.drift import DriftDetector


def run_drift_check():
    """
    Загружает референсные и текущие признаки,
    запускает детектор дрейфа и сохраняет результат.
    """
    print("=" * 50)
    print("Running Drift Detection...")
    print("=" * 50)

    data_dir = Path("data")
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    # Загрузка данных
    print("\nLoading feature sets...")
    try:
        reference_features = pd.read_parquet(data_dir / "reference_features.parquet")
        current_features = pd.read_parquet(data_dir / "features.parquet")
        print(f"Loaded reference features (v1): {reference_features.shape}")
        print(f"Loaded current features (v2): {current_features.shape}")
    except FileNotFoundError as e:
        print(f"Error loading feature files: {e}")
        return

    # Детекция дрейфа
    detector = DriftDetector(params_path="params.yaml", reports_path="reports")

    # Устанавливаем референс вручную, т.к. скрипт не знает о версиях
    detector.set_reference(reference_features, version="v1")

    drift_result = detector.detect_drift(current_features)

    print("\nDrift detection result:")
    print(json.dumps(drift_result, indent=4))

    # Сохранение метрик
    metrics_path = reports_dir / "drift_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(drift_result, f, indent=4)
    print(f"\nDrift metrics saved to {metrics_path}")

    # Проверяем, нужно ли переобучение
    if detector.should_retrain():
        print("\nDrift detected! Model retraining is recommended.")
    else:
        print("\nNo significant drift detected. Retraining is not required.")

    print("\n" + "=" * 50)
    print("Drift detection completed!")
    print("=" * 50)


if __name__ == "__main__":
    run_drift_check()
