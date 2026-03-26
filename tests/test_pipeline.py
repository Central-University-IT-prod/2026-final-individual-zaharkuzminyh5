#!/usr/bin/env python3
"""
Тест пайплайна на данных из /app/data/raw/v1.
"""

import sys
import json
from pathlib import Path
import pandas as pd

sys.path.insert(0, '/app/src')

from ml.features import FeatureBuilder
from ml.model import LookalikeModel

DATA_DIR = Path("/app/data/raw/v1")

def load_jsonl(name):
    """Загружает JSONL файл."""
    # Ищем файлы батчей
    batch_files = sorted(DATA_DIR.glob(f"{name}_batch_*.jsonl"))
    
    if not batch_files:
        # Пробуем committed
        committed_file = DATA_DIR / "committed" / f"{name}.jsonl"
        if committed_file.exists():
            batch_files = [committed_file]
        else:
            print(f"❌ File not found: {name}")
            return pd.DataFrame()
    
    records = []
    for bf in batch_files[:5]:  # Берём первые 5 батчей для теста
        with open(bf, 'r') as f:
            for line in f:
                records.append(json.loads(line))
    
    df = pd.DataFrame(records)
    print(f"✓ Loaded {name}: {len(df)} rows, columns: {list(df.columns)[:10]}")
    return df

def main():
    print("=" * 60)
    print("Тест пайплайна на данных /app/data/raw/v1")
    print("=" * 60)
    
    # Загружаем таблицы
    tables = {}
    for name in ["people", "segments", "transaction", "offer", "merchant", 
                 "financial_account", "offer_seens", "offer_activation", "offer_reward"]:
        tables[name] = load_jsonl(name)
    
    # Проверяем offer_activation и offer_reward
    print(f"\noffer_activation: {len(tables['offer_activation'])} rows")
    print(f"offer_reward: {len(tables['offer_reward'])} rows")
    
    if len(tables['offer_activation']) > 0:
        print(f"  Sample: {tables['offer_activation'].iloc[0].to_dict()}")
    
    if len(tables['offer_reward']) > 0:
        print(f"  Sample reward: {tables['offer_reward'].iloc[0].to_dict()}")
    
    # Строим маппинги
    merchant_offer_map = {}
    offer_merchant_map = {}
    
    if len(tables['offer']) > 0:
        for _, row in tables['offer'].iterrows():
            mid = int(float(row['merchant_id_offer']))
            oid = int(float(row['offer_id']))
            if mid not in merchant_offer_map:
                merchant_offer_map[mid] = []
            merchant_offer_map[mid].append(oid)
            offer_merchant_map[oid] = mid
    
    print(f"\nMerchants: {len(merchant_offer_map)}, Offers: {len(offer_merchant_map)}")
    
    # Reference date
    if len(tables['transaction']) > 0 and 'event_date' in tables['transaction'].columns:
        reference_date = tables['transaction']['event_date'].max()
        print(f"Reference date: {reference_date}")
    else:
        reference_date = "2025-06-10"
        print(f"Using default reference date: {reference_date}")
    
    # Строим фичи
    print("\nBuilding features...")
    feature_builder = FeatureBuilder("/app/params.yaml")
    features, interactions = feature_builder.build_features(tables, reference_date)
    
    print(f"\nFeatures shape: {features.shape}")
    print(f"Interactions shape: {len(interactions)}")
    
    if len(interactions) > 0:
        print(f"  Sample interactions: {interactions.iloc[0].to_dict()}")
        print(f"  Unique users: {interactions['user_id'].nunique()}")
        print(f"  Unique offers: {interactions['offer_id'].nunique()}")
        print(f"  interactions dtypes: {interactions.dtypes.to_dict()}")
    else:
        print("  ⚠️ Interactions EMPTY!")
    
    # Пробуем обучить модель
    print("\nTraining model...")
    model = LookalikeModel()
    
    try:
        train_metrics = model.fit(
            features=features,
            interactions=interactions,
            merchant_offer_map=merchant_offer_map,
            offer_merchant_map=offer_merchant_map,
            existing_customers={}
        )
        print(f"✓ Model trained! Metrics: {train_metrics}")
        
        # Тест predict
        if len(offer_merchant_map) > 0:
            first_offer = list(offer_merchant_map.keys())[0]
            first_merchant = offer_merchant_map[first_offer]
            
            print(f"\nTesting predict for merchant={first_merchant}, offer={first_offer}...")
            audience = model.predict(first_merchant, first_offer, top_n=10)
            print(f"  Audience size: {len(audience)}")
            if audience:
                print(f"  Top user: {audience[0]}")
        
        print("\n✅ TEST PASSED!")
        
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
