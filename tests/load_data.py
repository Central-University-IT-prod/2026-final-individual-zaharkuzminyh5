#!/usr/bin/env python3
"""
Скрипт загрузки данных v1 в сервис.
Минимальные зависимости - только requests.
"""

import os
import sys
import json
import requests
import csv
from pathlib import Path
from typing import Dict, List

# Маппинг таблиц
TABLE_MAPPING = {
    "people": "prod_clients.csv",
    "segments": "prizm_segments.csv",
    "transaction": "prod_financial_transaction.csv",
    "offer": "t_offer.csv",
    "merchant": "t_merchant.csv",
    "financial_account": "financial_account.csv",
    "offer_seens": "offer_seens.csv",
    "offer_activation": "offer_activation.csv",
    "offer_reward": "offer_reward.csv",
}

BASE_URL = "http://localhost:80"
BATCH_SIZE = 5000  # Записей в батче


def convert_and_send_batch(
    records: List[Dict],
    version: str,
    table: str,
    batch_id: int,
    total_batches: int
):
    """Отправляет батч."""
    payload = {
        "version": version,
        "table": table,
        "batch_id": batch_id,
        "total_batches": total_batches,
        "records": records
    }
    
    response = requests.post(
        f"{BASE_URL}/data/batch",
        json=payload,
        timeout=120
    )
    
    return response.status_code == 200, response.text


def load_csv_and_send(file_path: Path, version: str, table: str):
    """Читает CSV по строкам и отправляет батчами."""
    print(f"\n📊 Table: {table}")
    
    # Сначала считаем строки для total_batches
    with open(file_path, 'r', encoding='utf-8') as f:
        total_rows = sum(1 for _ in f) - 1  # минус заголовок
    
    total_batches = (total_rows + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"   Total rows: {total_rows}, Batches: {total_batches}")
    
    batch = []
    batch_id = 0
    sent_batches = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            batch.append(row)
            
            if len(batch) >= BATCH_SIZE:
                batch_id += 1
                success, msg = convert_and_send_batch(
                    batch, version, table, batch_id, total_batches
                )
                
                if success:
                    sent_batches += 1
                    print(f"  ✓ Batch {batch_id}/{total_batches} sent")
                else:
                    print(f"  ✗ Batch {batch_id}/{total_batches} failed: {msg[:100]}")
                    return False, sent_batches
                
                batch = []
    
    # Последний батч
    if batch:
        batch_id += 1
        success, msg = convert_and_send_batch(
            batch, version, table, batch_id, total_batches
        )
        
        if success:
            sent_batches += 1
            print(f"  ✓ Batch {batch_id}/{total_batches} sent")
        else:
            print(f"  ✗ Batch {batch_id}/{total_batches} failed: {msg[:100]}")
            return False, sent_batches
    
    return True, sent_batches


def load_data(version: str = "v1", data_dir: str = "/home/omniikc/projects/zaharkuzminyh5/ml/v1"):
    """Загружает все таблицы."""
    data_dir = Path(data_dir)
    
    print("=" * 60)
    print(f"Loading data version: {version}")
    print(f"Data directory: {data_dir}")
    print("=" * 60)
    
    # Проверка готовности сервиса
    try:
        response = requests.get(f"{BASE_URL}/ready", timeout=5)
        if response.status_code != 200:
            print(f"❌ Service not ready: {response.status_code}")
            return False
        print("✅ Service is ready")
    except Exception as e:
        print(f"❌ Cannot connect to service: {e}")
        return False
    
    # Загрузка таблиц
    for table_name, file_name in TABLE_MAPPING.items():
        file_path = data_dir / file_name
        
        if not file_path.exists():
            print(f"⚠️  File not found: {file_path}")
            continue
        
        success, batches = load_csv_and_send(file_path, version, table_name)
        
        if not success:
            print(f"❌ Failed to send table: {table_name}")
            return False
    
    # Отправка commit
    print("\n" + "=" * 60)
    print("Sending commit...")
    print("=" * 60)
    
    response = requests.post(
        f"{BASE_URL}/data/commit",
        json={"version": version},
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Commit accepted!")
        print(f"   Tables received: {result.get('tables_received', [])}")
    else:
        print(f"❌ Commit failed: {response.status_code}")
        print(f"   Response: {response.text}")
        return False
    
    return True


def check_status(max_wait: int = 600):
    """Проверяет статус пайплайна."""
    print("\n" + "=" * 60)
    print(f"Checking pipeline status (max {max_wait}s)...")
    print("=" * 60)
    
    import time
    start = time.time()
    
    while time.time() - start < max_wait:
        try:
            response = requests.get(f"{BASE_URL}/status", timeout=5)
            if response.status_code == 200:
                status = response.json()
                elapsed = int(time.time() - start)
                print(f"  [{elapsed}s] ready={status.get('ready')}, pipeline={status.get('pipeline_status')}, model={status.get('model_version')}")
                
                if status.get("pipeline_status") == "idle":
                    print(f"\n✅ Pipeline completed in {elapsed}s!")
                    return status
            else:
                print(f"  Error: {response.status_code}")
        except Exception as e:
            print(f"  Error: {e}")
        
        time.sleep(2)
    
    print(f"\n⚠️  Pipeline timeout after {max_wait}s")
    return None


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Load data into Look-a-Like service")
    parser.add_argument("--version", default="v1", help="Data version")
    parser.add_argument("--data-dir", default="/home/omniikc/projects/zaharkuzminyh5/ml/v1", help="Data directory")
    parser.add_argument("--check-status", action="store_true", help="Check pipeline status after commit")
    parser.add_argument("--max-wait", type=int, default=600, help="Max wait time for pipeline")
    
    args = parser.parse_args()
    
    success = load_data(args.version, args.data_dir)
    
    if success and args.check_status:
        status = check_status(args.max_wait)
        
        if status and status.get("ready"):
            print("\n" + "=" * 60)
            print("Testing endpoints...")
            print("=" * 60)
            
            # Model info
            response = requests.get(f"{BASE_URL}/model/info", timeout=5)
            print(f"\n/model/info: {response.status_code}")
            if response.status_code == 200:
                print(f"  {response.json()}")
            
            # Experiments
            response = requests.get(f"{BASE_URL}/experiments", timeout=5)
            print(f"\n/experiments: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"  Experiments: {len(data.get('experiments', []))}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
