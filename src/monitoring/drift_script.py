"""
Скрипт проверки дрейфа для DVC пайплайна.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.monitoring.drift import DriftDetector


def check_drift(params_path: str = "/app/params.yaml", data_path: str = "/app/data"):
    """
    Основная функция проверки дрейфа.
    """
    print("=" * 50)
    print("Checking data drift...")
    print("=" * 50)
    
    data_path = Path(data_path)
    reports_path = Path("/app/reports")
    reports_path.mkdir(parents=True, exist_ok=True)
    
    # Загружаем текущие признаки
    features_path = data_path / "features.parquet"
    
    if not features_path.exists():
        print("Features file not found!")
        result = {
            "drift_detected": False,
            "drift_score": 0.0,
            "action_taken": "skipped",
            "message": "Features file not found"
        }
        
        with open(reports_path / "drift_report.json", 'w') as f:
            json.dump(result, f, indent=2)
        
        return result
    
    current_features = pd.read_parquet(features_path)
    print(f"Loaded current features: {current_features.shape}")
    
    # Загружаем референсные признаки
    ref_path = data_path / "reference_features.parquet"
    
    detector = DriftDetector(params_path, str(reports_path))
    
    if ref_path.exists():
        reference_features = pd.read_parquet(ref_path)
        print(f"Loaded reference features: {reference_features.shape}")
        
        # Детектируем дрейф
        drift_result = detector.detect_drift(current_features, reference_features)
    else:
        print("No reference features found. Saving current as reference.")
        detector.save_reference(current_features, str(ref_path))
        
        drift_result = {
            "drift_detected": False,
            "drift_score": 0.0,
            "action_taken": "none",
            "message": "Reference features saved"
        }
    
    # Сохраняем отчёт
    report_file = reports_path / "drift_report.json"
    with open(report_file, 'w') as f:
        json.dump(drift_result, f, indent=2, default=str)
    
    # Сохраняем метрики для DVC
    metrics = {
        "drift_detected": drift_result.get("drift_detected", False),
        "drift_score": drift_result.get("drift_score", 0.0),
        "drift_share": drift_result.get("drift_share", 0.0),
        "timestamp": datetime.now().isoformat()
    }
    
    metrics_file = reports_path / "drift_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nDrift detected: {drift_result.get('drift_detected', False)}")
    print(f"Drift score: {drift_result.get('drift_score', 0.0)}")
    print(f"Action: {drift_result.get('action_taken', 'none')}")
    
    print("\n" + "=" * 50)
    print("Drift check completed!")
    print("=" * 50)
    
    return drift_result


if __name__ == "__main__":
    params_path = sys.argv[1] if len(sys.argv) > 1 else "/app/params.yaml"
    data_path = sys.argv[2] if len(sys.argv) > 2 else "/app/data"
    check_drift(params_path, data_path)
