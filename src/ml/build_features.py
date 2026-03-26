"""
Скрипт построения признаков для DVC пайплайна.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader_impl import DataLoader
from ml.features import FeatureBuilder


def build_features(
    params_path: str = "params.yaml", data_path: str = "/app/data/v1_jsonl"
):
    """
    Основная функция построения признаков.
    """
    print("=" * 50)
    print("Building features...")
    print("=" * 50)

    data_path = Path(data_path)

    # Загружаем таблицы
    print(f"\nLoading data from {data_path}...")
    tables = {}

    table_files = {
        "people": "prod_clients.jsonl",
        "segments": "prizm_segments.jsonl",
        "transaction": "prod_financial_transaction.jsonl",
        "offer": "t_offer.jsonl",
        "merchant": "t_merchant.jsonl",
        "financial_account": "financial_account.jsonl",
        "offer_seens": "offer_seens.jsonl",
        "offer_activation": "offer_activation.jsonl",
        "offer_reward": "offer_reward.jsonl",
        "receipts": "receipts.jsonl",
    }

    for table_name, file_name in table_files.items():
        file_path = data_path / file_name

        if file_path.exists():
            records = []
            with open(file_path, "r") as f:
                for line in f:
                    records.append(json.loads(line))
            tables[table_name] = pd.DataFrame(records)
            print(f"Loaded {table_name}: {len(tables[table_name])} records")
        else:
            print(f"Warning: {file_path} not found. Skipping.")

    if len(tables) == 0:
        print("No data loaded. Exiting.")
        return

    # Reference date
    reference_date = "2026-02-18"
    if "transaction" in tables and len(tables["transaction"]) > 0:
        reference_date = tables["transaction"]["event_date"].max()

    # Построение признаков
    print("\nBuilding features...")
    feature_builder = FeatureBuilder(params_path)
    features, interactions = feature_builder.build_features(tables, reference_date)

    print(f"Features shape: {features.shape}")
    print(f"Interactions shape: {len(interactions)}")

    # Сохранение
    output_dir = Path("data")  # Save features to data/features.parquet
    feature_builder.save_features(features, interactions, output_dir)
    print(f"\nSaved features to {output_dir}")

    print("\n" + "=" * 50)
    print("Feature building completed!")
    print("=" * 50)


if __name__ == "__main__":
    params_path = sys.argv[1] if len(sys.argv) > 1 else "params.yaml"
    data_path = sys.argv[2] if len(sys.argv) > 2 else "/app/data/v1_jsonl"
    build_features(params_path, data_path)
