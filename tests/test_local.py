"""
Скрипт для локального тестирования на данных v1/v2.
Извлекает данные из архивов и подготавливает для загрузки через API.
"""

import os
import sys
import json
import zipfile
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np


# Маппинг логических имён таблиц в файлы
TABLE_FILES = {
    "people": "prod_clients.csv",
    "segments": "prizm_segments.csv",
    "transaction": "prod_financial_transaction.csv",
    "offer": "t_offer.csv",
    "merchant": "t_merchant.csv",
    "financial_account": "financial_account.csv",
    "offer_seens": "offer_seens.csv",
    "offer_activation": "offer_activation.csv",
    "offer_reward": "offer_reward.csv",
    "receipts": "receipts.csv"
}


def extract_version(archive_path: str, version: str, output_dir: str):
    """Извлекает версию из архива."""
    archive_path = Path(archive_path)
    output_dir = Path(output_dir)
    
    if not archive_path.exists():
        print(f"Archive not found: {archive_path}")
        return
    
    print(f"Extracting {version} from {archive_path}...")
    
    with zipfile.ZipFile(archive_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    
    print(f"Extracted to {output_dir / version}")


def prepare_batch_data(
    data_dir: str,
    version: str,
    output_dir: str
) -> Dict[str, List[Dict]]:
    """
    Подготавливает данные для отправки через API батчами.
    """
    data_dir = Path(data_dir) / version
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    batches = {}
    
    for table_name, file_name in TABLE_FILES.items():
        file_path = data_dir / file_name
        
        if not file_path.exists():
            print(f"File not found: {file_path}")
            continue
        
        print(f"Processing {table_name}...")
        
        # Читаем CSV
        df = pd.read_csv(file_path)
        
        # Разбиваем на батчи по 10000 записей
        batch_size = 10000
        n_batches = (len(df) + batch_size - 1) // batch_size
        
        table_batches = []
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(df))
            
            batch_df = df.iloc[start_idx:end_idx]
            
            # Конвертируем в JSON records
            records = batch_df.to_dict(orient='records')
            
            # Обрабатываем NaN и специальные значения
            for record in records:
                for key, value in record.items():
                    if pd.isna(value):
                        record[key] = None
                    elif isinstance(value, (np.int64, np.int32)):
                        record[key] = int(value)
                    elif isinstance(value, (np.float64, np.float32)):
                        record[key] = float(value)
            
            batch_data = {
                "version": version,
                "table": table_name,
                "batch_id": i + 1,
                "total_batches": n_batches,
                "records": records
            }
            
            # Сохраняем батч
            batch_file = output_dir / f"{table_name}_batch_{i + 1}.json"
            with open(batch_file, 'w') as f:
                json.dump(batch_data, f)
            
            table_batches.append(batch_file)
        
        batches[table_name] = table_batches
        print(f"  Created {n_batches} batches ({len(df)} records)")
    
    # Сохраняем манифест
    manifest = {
        "version": version,
        "tables": {k: len(v) for k, v in batches.items()},
        "batch_files": {k: [str(f) for f in v] for k, v in batches.items()}
    }
    
    manifest_file = output_dir / "manifest.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\nManifest saved to {manifest_file}")
    
    return batches


def send_to_api(base_url: str, data_dir: str, version: str):
    """Отправляет данные на API."""
    import requests
    
    data_dir = Path(data_dir)
    manifest_file = data_dir / "manifest.json"
    
    if not manifest_file.exists():
        print(f"Manifest not found: {manifest_file}")
        return
    
    with open(manifest_file, 'r') as f:
        manifest = json.load(f)
    
    print(f"\nSending {version} data to {base_url}...")
    
    # Отправляем батчи
    for table_name, n_batches in manifest["tables"].items():
        print(f"\nSending {table_name} ({n_batches} batches)...")
        
        for i in range(1, n_batches + 1):
            batch_file = data_dir / f"{table_name}_batch_{i}.json"
            
            with open(batch_file, 'r') as f:
                batch_data = json.load(f)
            
            response = requests.post(
                f"{base_url}/data/batch",
                json=batch_data,
                timeout=30
            )
            
            if response.status_code == 200:
                print(f"  Batch {i}/{n_batches}: OK")
            else:
                print(f"  Batch {i}/{n_batches}: FAILED ({response.status_code})")
    
    # Отправляем commit
    print(f"\nSending commit for {version}...")
    response = requests.post(
        f"{base_url}/data/commit",
        json={"version": version},
        timeout=30
    )
    
    if response.status_code == 200:
        print(f"Commit: OK")
        print(f"Response: {response.json()}")
    else:
        print(f"Commit: FAILED ({response.status_code})")


def main():
    """Основная функция."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test script for Look-a-Like service")
    parser.add_argument("--action", choices=["extract", "prepare", "send", "all"],
                        default="all", help="Action to perform")
    parser.add_argument("--version", default="v1", help="Version to process")
    parser.add_argument("--archive", default="/home/omniikc/projects/zaharkuzminyh5/ml/v1.zip",
                        help="Path to archive")
    parser.add_argument("--output", default="/tmp/lookalike_data",
                        help="Output directory")
    parser.add_argument("--api-url", default="http://localhost:80",
                        help="API base URL")
    
    args = parser.parse_args()
    
    if args.action in ["extract", "all"]:
        extract_version(args.archive, args.version, args.output)
    
    if args.action in ["prepare", "all"]:
        prepare_batch_data(args.output, args.version, args.output)
    
    if args.action in ["send", "all"]:
        send_to_api(args.api_url, Path(args.output) / args.version, args.version)


if __name__ == "__main__":
    main()
