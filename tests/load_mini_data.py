#!/usr/bin/env python3
"""
Скрипт загрузки мини-данных для быстрого тестирования.
"""

import requests
import json
from pathlib import Path

BASE_URL = "http://localhost:80"

# Мини-данные для теста
MINI_DATA = {
    "people": [
        {"user_id": 1, "age_bucket": "25", "gender_cd": "M", "region": "МОСКВА", "last_activity_day": "2025-06-01"},
        {"user_id": 2, "age_bucket": "30", "gender_cd": "F", "region": "МОСКВА", "last_activity_day": "2025-06-02"},
        {"user_id": 3, "age_bucket": "35", "gender_cd": "M", "region": "СПБ", "last_activity_day": "2025-06-03"},
        {"user_id": 4, "age_bucket": "25", "gender_cd": "F", "region": "СПБ", "last_activity_day": "2025-06-04"},
        {"user_id": 5, "age_bucket": "40", "gender_cd": "M", "region": "МОСКВА", "last_activity_day": "2025-06-05"},
    ],
    "segments": [
        {"user_id": 1, "segment": "m_02 (35)", "region_size": "urban", "auto": 0.0, "traveler": 1.0, "entrepreneur": 0.0, "vip_status": "not_vip"},
        {"user_id": 2, "segment": "m_03 (28)", "region_size": "urban", "auto": 0.0, "traveler": 0.0, "entrepreneur": 0.0, "vip_status": "not_vip"},
        {"user_id": 3, "segment": "t_02 (42)", "region_size": "town", "auto": 1.0, "traveler": 0.0, "entrepreneur": 0.0, "vip_status": "not_vip"},
        {"user_id": 4, "segment": "u_05 (31)", "region_size": "urban", "auto": 0.0, "traveler": 1.0, "entrepreneur": 1.0, "vip_status": "not_vip"},
        {"user_id": 5, "segment": "m_01 (55)", "region_size": "urban", "auto": 1.0, "traveler": 0.0, "entrepreneur": 1.0, "vip_status": "vip"},
    ],
    "merchant": [
        {"merchant_id_offer": 75, "merchant_status": "ACT", "brand_dk": 18601},
        {"merchant_id_offer": 110, "merchant_status": "ACT", "brand_dk": 18937},
    ],
    "offer": [
        {"offer_id": 42, "merchant_id_offer": 75, "start_date": "2025-06-03", "end_date": "2025-06-30", "offer_text": "Тестовый оффер 1"},
        {"offer_id": 99, "merchant_id_offer": 110, "start_date": "2025-07-01", "end_date": "2025-07-31", "offer_text": "Тестовый оффер 2"},
    ],
    "transaction": [
        {"transaction_id": 1, "user_id": 1, "merchant_id_tx": 75, "event_date": "2025-06-01", "amount_bucket": "1k+", "online_transaction_flg": "N", "brand_dk": 18601},
        {"transaction_id": 2, "user_id": 1, "merchant_id_tx": 75, "event_date": "2025-06-02", "amount_bucket": "5k+", "online_transaction_flg": "Y", "brand_dk": 18601},
        {"transaction_id": 3, "user_id": 2, "merchant_id_tx": 110, "event_date": "2025-06-01", "amount_bucket": "1k+", "online_transaction_flg": "N", "brand_dk": 18937},
        {"transaction_id": 4, "user_id": 3, "merchant_id_tx": 75, "event_date": "2025-06-03", "amount_bucket": "10k+", "online_transaction_flg": "N", "brand_dk": 18601},
        {"transaction_id": 5, "user_id": 4, "merchant_id_tx": 110, "event_date": "2025-06-04", "amount_bucket": "1k+", "online_transaction_flg": "Y", "brand_dk": 18937},
        {"transaction_id": 6, "user_id": 5, "merchant_id_tx": 75, "event_date": "2025-06-05", "amount_bucket": "5k+", "online_transaction_flg": "N", "brand_dk": 18601},
    ],
    "financial_account": [
        {"user_id": 1, "product_cd": "CARD", "open_month": "2024-01", "close_month": "", "account_status_cd": "ACT"},
        {"user_id": 2, "product_cd": "CARD", "open_month": "2024-02", "close_month": "", "account_status_cd": "ACT"},
        {"user_id": 3, "product_cd": "SAVINGS", "open_month": "2024-03", "close_month": "", "account_status_cd": "ACT"},
        {"user_id": 4, "product_cd": "CARD", "open_month": "2024-04", "close_month": "", "account_status_cd": "ACT"},
        {"user_id": 5, "product_cd": "PREMIUM", "open_month": "2024-05", "close_month": "", "account_status_cd": "ACT"},
    ],
    "offer_seens": [
        {"user_id": 1, "offer_id": 42, "start_date": "2025-06-03", "end_date": "2025-06-30"},
        {"user_id": 2, "offer_id": 99, "start_date": "2025-07-01", "end_date": "2025-07-31"},
        {"user_id": 3, "offer_id": 42, "start_date": "2025-06-03", "end_date": "2025-06-30"},
    ],
    "offer_activation": [
        {"user_id": 1, "offer_id": 42, "activation_date": "2025-06-05"},
        {"user_id": 3, "offer_id": 42, "activation_date": "2025-06-10"},
    ],
    "offer_reward": [
        {"user_id": 1, "offer_id": 42, "event_date": "2025-06-15", "reward_amt": 100},
    ],
    "receipts": [
        {"user_id": 1, "date_operated": "2025-06-01", "category_name": "PRODUCT", "items_count": 5, "items_cost": 1500},
        {"user_id": 2, "date_operated": "2025-06-02", "category_name": "FOOD", "items_count": 3, "items_cost": 800},
        {"user_id": 3, "date_operated": "2025-06-03", "category_name": "PRODUCT", "items_count": 2, "items_cost": 2000},
    ],
}


def send_table(table_name: str, records: list, version: str = "mini"):
    """Отправляет таблицу."""
    payload = {
        "version": version,
        "table": table_name,
        "batch_id": 1,
        "total_batches": 1,
        "records": records
    }
    
    response = requests.post(f"{BASE_URL}/data/batch", json=payload, timeout=30)
    
    if response.status_code == 200:
        print(f"  ✓ {table_name}: {len(records)} records")
        return True
    else:
        print(f"  ✗ {table_name}: {response.status_code} - {response.text[:100]}")
        return False


def main():
    print("=" * 60)
    print("Loading mini data for testing")
    print("=" * 60)
    
    # Проверка готовности
    r = requests.get(f"{BASE_URL}/ready", timeout=5)
    if r.status_code != 200:
        print(f"Service not ready: {r.status_code}")
        return
    
    print("Service is ready\n")
    
    # Отправка таблиц
    for table_name, records in MINI_DATA.items():
        send_table(table_name, records)
    
    # Commit
    print("\nSending commit...")
    response = requests.post(
        f"{BASE_URL}/data/commit",
        json={"version": "mini"},
        timeout=30
    )
    
    if response.status_code == 200:
        print(f"✅ Commit accepted!")
        print(f"   Tables: {response.json().get('tables_received', [])}")
    else:
        print(f"❌ Commit failed: {response.status_code}")
        return
    
    # Ожидание пайплайна
    print("\nWaiting for pipeline...")
    import time
    for i in range(60):
        r = requests.get(f"{BASE_URL}/status", timeout=5)
        s = r.json()
        
        if s.get("pipeline_status") == "idle":
            print(f"\n✅ Pipeline completed in {i*2}s!")
            print(f"   Status: {s}")
            break
        
        if i % 10 == 0:
            print(f"  [{i*2}s] pipeline={s.get('pipeline_status')}")
        
        time.sleep(2)
    else:
        print("⚠️  Pipeline timeout")
    
    # Проверка модели
    print("\nChecking model...")
    r = requests.get(f"{BASE_URL}/model/info", timeout=5)
    print(f"Model info: {r.status_code}")
    if r.status_code == 200:
        print(r.json())
    
    # Тест lookalike
    print("\nTesting /lookalike...")
    r = requests.post(
        f"{BASE_URL}/lookalike",
        json={"merchant_id": 75, "offer_id": 42, "top_n": 5},
        timeout=30
    )
    print(f"Lookalike: {r.status_code}")
    if r.status_code == 200:
        result = r.json()
        print(f"  Audience size: {result.get('audience_size')}")
        print(f"  Model version: {result.get('model_version')}")
        if result.get('audience'):
            print(f"  Top user: {result['audience'][0]}")
        if result.get('reasons'):
            print(f"  Reasons: {result['reasons'][:2]}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
