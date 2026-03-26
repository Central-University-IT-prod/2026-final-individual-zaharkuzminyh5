"""
Модуль детекции дрейфа данных с использованием Evidently.
Сравнивает распределения признаков между референсной и текущей версией.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


class DriftDetector:
    """
    Детектор дрейфа данных на основе Evidently.
    """

    def __init__(
        self, params_path: str = "/app/params.yaml", reports_path: str = "/app/reports"
    ):
        with open(params_path, "r") as f:
            self.params = yaml.safe_load(f)

        self.drift_params = self.params.get("drift", {})
        self.threshold = self.drift_params.get("threshold", 0.5)
        self.reference_version = self.drift_params.get("reference_version", "v1")

        self.reports_path = Path(reports_path)
        self.reports_path.mkdir(parents=True, exist_ok=True)

        self._reference_features: Optional[pd.DataFrame] = None
        self._last_drift_result: Optional[Dict] = None

    def set_reference(self, features: pd.DataFrame, version: str = "v1"):
        """Устанавливает референсные данные для сравнения."""
        self._reference_features = features.copy()
        self.reference_version = version

    def load_reference(
        self, features_path: str = "/app/data/reference_features.parquet"
    ) -> bool:
        """Загружает референсные признаки из файла."""
        path = Path(features_path)
        if path.exists():
            self._reference_features = pd.read_parquet(path)
            return True
        return False

    def save_reference(
        self,
        features: pd.DataFrame,
        features_path: str = "/app/data/reference_features.parquet",
    ):
        """Сохраняет референсные признаки."""
        path = Path(features_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(path, index=False)
        self._reference_features = features.copy()

    def detect_drift(
        self,
        current_features: pd.DataFrame,
        reference_features: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Детектирует дрейф между референсными и текущими данными."""
        # Используем предоставленные или сохранённые референсные данные
        if reference_features is not None:
            ref_features = reference_features.copy()
        elif self._reference_features is not None:
            ref_features = self._reference_features
        else:
            return {
                "drift_detected": False,
                "drift_score": 0.0,
                "action_taken": "skipped",
                "message": "No reference data available",
            }

        # Выбираем числовые колонки для сравнения
        numeric_cols = current_features.select_dtypes(
            include=[np.number]
        ).columns.tolist()

        # Фильтруем колонки, которые есть в обоих датасетах
        common_cols = [
            col
            for col in numeric_cols
            if col in ref_features.columns and col not in ["user_id"]
        ]

        if len(common_cols) == 0:
            return {
                "drift_detected": False,
                "drift_score": 0.0,
                "action_taken": "skipped",
                "message": "No common numeric columns for drift detection",
            }

        # Ограничиваем количество колонок для скорости
        # common_cols = common_cols[:50]

        # Вычисляем PSI (Population Stability Index) для каждой колонки
        drift_scores = {}
        drifted_cols = []

        for col in common_cols:
            # Конвертируем в numeric если нужно
            ref_col = pd.to_numeric(ref_features[col], errors="coerce").dropna()
            curr_col = pd.to_numeric(current_features[col], errors="coerce").dropna()

            psi_score = self._calculate_psi(ref_col, curr_col)
            drift_scores[col] = psi_score

            if psi_score > self.threshold:
                drifted_cols.append(col)

        # Агрегируем дрейф
        avg_drift_score = np.mean(list(drift_scores.values())) if drift_scores else 0.0
        drift_share = len(drifted_cols) / len(common_cols) if common_cols else 0.0

        # Решение о дрейфе
        drift_detected = drift_share > 0.01 or avg_drift_score > 0.1

        # Генерируем отчёт
        drift_result = {
            "drift_detected": bool(drift_detected),
            "drift_score": round(float(avg_drift_score), 4),
            "drift_share": round(float(drift_share), 4),
            "threshold": self.threshold,
            "drifted_columns": drifted_cols,
            "column_scores": {k: round(v, 4) for k, v in drift_scores.items()},
            "n_columns_total": len(common_cols),
            "n_columns_drifted": len(drifted_cols),
            "reference_version": self.reference_version,
            "timestamp": datetime.now().isoformat(),
        }

        self._last_drift_result = drift_result

        return drift_result

    def _calculate_psi(
        self, reference: pd.Series, current: pd.Series, n_buckets: int = 10
    ) -> float:
        """
        Вычисляет Population Stability Index (PSI).

        PSI < 0.1: нет дрейфа
        0.1 <= PSI < 0.2: небольшой дрейф
        PSI >= 0.2: значительный дрейф
        """
        if len(reference) == 0 or len(current) == 0:
            return 0.0

        # Создаём бакеты на основе референсных данных
        breakpoints = np.percentile(reference, np.linspace(0, 100, n_buckets + 1))
        breakpoints = np.unique(breakpoints)

        if len(breakpoints) < 2:
            return 0.0

        # Распределяем данные по бакетам
        ref_counts = np.histogram(reference, bins=breakpoints)[0]
        curr_counts = np.histogram(current, bins=breakpoints)[0]

        # Нормализуем до процентов
        ref_percents = (ref_counts + 1) / (len(reference) + n_buckets)
        curr_percents = (curr_counts + 1) / (len(current) + n_buckets)

        # Вычисляем PSI
        psi = np.sum(
            (curr_percents - ref_percents) * np.log(curr_percents / ref_percents)
        )

        return float(psi)

    def get_drift_summary(self) -> Dict[str, Any]:
        """Возвращает сводку о дрейфе для API."""
        if self._last_drift_result is None:
            return {"drift_detected": False, "drift_score": 0.0, "action_taken": "none"}

        result = self._last_drift_result
        action = "retrained" if result["drift_detected"] else "none"

        return {
            "drift_detected": result["drift_detected"],
            "drift_score": result["drift_score"],
            "action_taken": action,
            "drift_share": result.get("drift_share", 0.0),
            "drifted_columns": result.get("drifted_columns", [])[:5],
            "threshold": result.get("threshold", self.threshold),
            "threshold": result.get("threshold", self.threshold),
        }

    def should_retrain(self) -> bool:
        """Проверяет, нужно ли переобучать модель."""
        if self._last_drift_result is None:
            return False

        return self._last_drift_result.get("drift_detected", False)


def get_drift_detector(params_path: str = "/app/params.yaml") -> DriftDetector:
    """Возвращает singleton детектора дрейфа."""
    return DriftDetector(params_path=params_path)


if __name__ == "__main__":
    print("=" * 50)
    print("Checking data drift...")
    print("=" * 50)

    reports_dir = Path("/app/reports")
    reports_dir.mkdir(exist_ok=True)

    # Загрузка текущих признаков
    print("\nLoading current features...")
    try:
        current_features = pd.read_parquet("/app/data/features.parquet")
        print(f"Loaded current features: {current_features.shape}")
    except FileNotFoundError:
        print("Error: Current features not found.")
        exit(1)

    # Инициализация детектора
    detector = get_drift_detector()

    # Загрузка референсных признаков
    if not detector.load_reference():
        print("Reference features not found. Saving current as reference.")
        detector.save_reference(current_features)

    # Детекция дрейфа
    print("\nDetecting drift...")
    drift_result = detector.detect_drift(current_features)

    # Сохранение отчета
    report_path = reports_dir / "drift_report.json"
    with open(report_path, "w") as f:
        json.dump(drift_result, f, indent=4)
    print(f"Drift report saved to {report_path}")

    # Сохранение метрик для DVC
    metrics = {
        "drift_detected": drift_result["drift_detected"],
        "drift_score": drift_result["drift_score"],
        "drift_share": drift_result["drift_share"],
        "n_columns_drifted": drift_result["n_columns_drifted"],
        "timestamp": drift_result["timestamp"],
    }
    metrics_path = reports_dir / "drift_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Drift metrics saved to {metrics_path}")

    print("\n" + "=" * 50)
    print("Drift check completed!")
    print("=" * 50)
