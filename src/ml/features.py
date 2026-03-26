"""
Модуль построения признаков (feature engineering) для Look-a-Like модели.
Создаёт агрегированные признаки пользователей на основе транзакций, сегментов и чеков.
"""

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


class FeatureBuilder:
    """
    Построитель признаков для модели look-a-like.
    Создаёт признаки на основе:
    - Демографии пользователей (people)
    - Психографических сегментов (segments)
    - Транзакционной истории (transaction)
    - Чеков (receipts)
    - Финансовых счетов (financial_account)
    """

    def __init__(self, params_path: str = "params.yaml"):
        with open(params_path, "r") as f:
            self.params = yaml.safe_load(f)

        self.feature_params = self.params.get("features", {})
        self.tx_windows = self.feature_params.get("tx_windows", [7, 30, 90])
        self.receipt_windows = self.feature_params.get("receipt_windows", [30, 90])

    def build_features(
        self, tables: Dict[str, pd.DataFrame], reference_date: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Строит признаки для всех пользователей.

        Args:
            tables: Словарь с таблицами данных
            reference_date: Дата, относительно которой считаются окна

        Returns:
            features: DataFrame с признаками пользователей
            interactions: DataFrame с взаимодействиями user-offer для обучения
        """
        # Устанавливаем reference date
        if reference_date is None:
            # Берём максимальную дату из транзакций
            if "transaction" in tables and len(tables["transaction"]) > 0:
                reference_date = tables["transaction"]["event_date"].max()
            else:
                reference_date = datetime.now().strftime("%Y-%m-%d")

        reference_date = pd.to_datetime(reference_date)

        # Базовые признаки из people
        features = self._build_base_features(tables)

        # Признаки из сегментов
        features = self._merge_segment_features(features, tables)

        # Транзакционные признаки
        features = self._build_transaction_features(features, tables, reference_date)

        # Признаки из чеков
        features = self._build_receipt_features(features, tables, reference_date)

        # Признаки из финансовых счетов
        features = self._build_account_features(features, tables)

        # Признаки взаимодействия с офферами
        features = self._build_offer_features(features, tables, reference_date)

        # Строим interactions для обучения
        interactions = self._build_interactions(tables, reference_date)

        return features, interactions

    def _build_base_features(self, tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Базовые признаки из таблицы people."""
        if "people" not in tables or len(tables["people"]) == 0:
            return pd.DataFrame()

        people = tables["people"].copy()

        # Кодирование пола
        people["gender_encoded"] = people["gender_cd"].map({"M": 1, "F": 0}).fillna(-1)

        # Кодирование age_bucket
        age_mapping = {
            "<18": 0,
            "15": 0,
            "18-24": 1,
            "25": 1,
            "25-34": 2,
            "30": 2,
            "35": 2,
            "35-44": 3,
            "40": 3,
            "45": 3,
            "45-54": 4,
            "50": 4,
            "55": 4,
            "55+": 5,
            "60": 5,
            "65": 5,
        }
        people["age_encoded"] = people["age_bucket"].apply(
            lambda x: age_mapping.get(str(x), -1)
        )

        # One-hot encoding для региона
        if "region" in people.columns:
            region_dummies = pd.get_dummies(people["region"], prefix="region")
            people = pd.concat([people, region_dummies], axis=1)

        # Дата последней активности
        if "last_activity_day" in people.columns:
            people["last_activity_day"] = pd.to_datetime(
                people["last_activity_day"], errors="coerce"
            )
            ref_date = pd.to_datetime("2026-02-18")
            days_diff = (ref_date - people["last_activity_day"]).dt.days
            people["days_since_activity"] = days_diff.fillna(999).astype("int32")

        return people

    def _merge_segment_features(
        self, features: pd.DataFrame, tables: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Добавляет признаки из сегментов."""
        if "segments" not in tables or len(tables["segments"]) == 0:
            return features

        if len(features) == 0:
            return features

        segments = tables["segments"].copy()

        # Извлекаем сегмент без номера в скобках
        segments["segment_code"] = segments["segment"].str.extract(
            r"^([a-z]_\d{2})", expand=False
        )
        segments["segment_rank"] = pd.to_numeric(
            segments["segment"].str.extract(r"\((\d+)\)", expand=False), errors="coerce"
        ).fillna(0)

        # Агрегируем сегменты по пользователю (берём первый)
        segment_agg = (
            segments.groupby("user_id")
            .agg(
                {
                    "segment_code": "first",
                    "segment_rank": "mean",
                    "region_size": "first",
                    "auto": lambda x: pd.to_numeric(x, errors="coerce").mean(),
                    "traveler": lambda x: pd.to_numeric(x, errors="coerce").mean(),
                    "entrepreneur": lambda x: pd.to_numeric(x, errors="coerce").mean(),
                    "vip_status": "first",
                }
            )
            .reset_index()
        )

        # One-hot encoding для сегментов
        segment_dummies = pd.get_dummies(segment_agg["segment_code"], prefix="seg")
        segment_agg = pd.concat([segment_agg, segment_dummies], axis=1)

        # One-hot encoding для region_size
        region_dummies = pd.get_dummies(
            segment_agg["region_size"], prefix="region_type"
        )
        segment_agg = pd.concat([segment_agg, region_dummies], axis=1)

        # Мердж
        features = features.merge(segment_agg, on="user_id", how="left")

        return features

    def _build_transaction_features(
        self,
        features: pd.DataFrame,
        tables: Dict[str, pd.DataFrame],
        reference_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """Строит признаки на основе транзакций."""
        if "transaction" not in tables or len(tables["transaction"]) == 0:
            return features

        if len(features) == 0:
            return features

        tx = tables["transaction"].copy()
        tx["event_date"] = pd.to_datetime(tx["event_date"], errors="coerce")

        # Кодирование онлайн транзакций
        tx["online_flag"] = tx["online_transaction_flg"].isin(["Y", "y", 1]).astype(int)

        # Кодирование amount_bucket в числа
        amount_mapping = {
            "<1k": 500,
            "1k+": 1000,
            "5k+": 5000,
            "10k+": 10000,
            "20k+": 20000,
            "50k+": 50000,
            "100k+": 100000,
        }
        tx["amount_numeric"] = tx["amount_bucket"].map(amount_mapping).fillna(0)

        # Агрегаты по окнам
        for window in self.tx_windows:
            window_start = reference_date - timedelta(days=window)
            tx_window = tx[tx["event_date"] >= window_start]

            if len(tx_window) == 0:
                features[f"tx_count_{window}d"] = 0
                features[f"tx_amount_sum_{window}d"] = 0.0
                features[f"tx_amount_avg_{window}d"] = 0.0
                features[f"online_share_{window}d"] = 0.0
                features[f"unique_merchants_{window}d"] = 0
                continue

            agg = (
                tx_window.groupby("user_id")
                .agg(
                    {
                        "transaction_id": "count",
                        "amount_numeric": ["sum", "mean"],
                        "online_flag": "mean",
                        "merchant_id_tx": "nunique",
                    }
                )
                .reset_index()
            )

            agg.columns = [
                "user_id",
                f"tx_count_{window}d",
                f"tx_amount_sum_{window}d",
                f"tx_amount_avg_{window}d",
                f"online_share_{window}d",
                f"unique_merchants_{window}d",
            ]

            features = features.merge(agg, on="user_id", how="left")

        # Общие агрегаты за всё время
        tx_agg = (
            tx.groupby("user_id")
            .agg(
                {
                    "transaction_id": "count",
                    "amount_numeric": ["sum", "mean", "std"],
                    "online_flag": "mean",
                    "merchant_id_tx": "nunique",
                    "brand_dk": "nunique",
                }
            )
            .reset_index()
        )

        tx_agg.columns = [
            "user_id",
            "tx_count_total",
            "tx_amount_sum_total",
            "tx_amount_avg_total",
            "tx_amount_std_total",
            "online_share_total",
            "unique_merchants_total",
            "unique_brands_total",
        ]

        features = features.merge(tx_agg, on="user_id", how="left")

        # Заполняем NaN нулями для числовых признаков
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        features[numeric_cols] = features[numeric_cols].fillna(0)

        return features

    def _build_receipt_features(
        self,
        features: pd.DataFrame,
        tables: Dict[str, pd.DataFrame],
        reference_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """Строит признаки на основе чеков."""
        if "receipts" not in tables or len(tables["receipts"]) == 0:
            return features

        if len(features) == 0:
            return features

        receipts = tables["receipts"].copy()
        receipts["date_operated"] = pd.to_datetime(
            receipts["date_operated"], errors="coerce"
        )

        for window in self.receipt_windows:
            window_start = reference_date - timedelta(days=window)
            rcpt_window = receipts[receipts["date_operated"] >= window_start]

            if len(rcpt_window) == 0:
                features[f"receipts_count_{window}d"] = 0
                features[f"receipts_items_{window}d"] = 0.0
                features[f"receipts_cost_{window}d"] = 0.0
                continue

            agg = (
                rcpt_window.groupby("user_id")
                .agg(
                    {
                        "date_operated": "count",
                        "items_count": "sum",
                        "items_cost": "sum",
                    }
                )
                .reset_index()
            )

            agg.columns = [
                "user_id",
                f"receipts_count_{window}d",
                f"receipts_items_{window}d",
                f"receipts_cost_{window}d",
            ]

            features = features.merge(agg, on="user_id", how="left")

        # Категории покупок
        if "category_name" in receipts.columns:
            category_counts = (
                receipts.groupby(["user_id", "category_name"])
                .size()
                .unstack(fill_value=0)
            )
            category_counts.columns = [f"cat_{c}" for c in category_counts.columns]
            category_counts = category_counts.reset_index()
            features = features.merge(category_counts, on="user_id", how="left")

        # Заполняем NaN
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        features[numeric_cols] = features[numeric_cols].fillna(0)

        return features

    def _build_account_features(
        self, features: pd.DataFrame, tables: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Добавляет признаки из финансовых счетов."""
        if "financial_account" not in tables or len(tables["financial_account"]) == 0:
            return features

        if len(features) == 0:
            return features

        accounts = tables["financial_account"].copy()

        # Агрегаты по счетам
        account_agg = (
            accounts.groupby("user_id")
            .agg(
                {
                    "product_cd": "nunique",
                    "account_status_cd": lambda x: (x == "ACT").sum(),
                }
            )
            .reset_index()
        )

        account_agg.columns = ["user_id", "num_products", "active_accounts"]

        features = features.merge(account_agg, on="user_id", how="left")
        features["num_products"] = features["num_products"].fillna(0)
        features["active_accounts"] = features["active_accounts"].fillna(0)

        return features

    def _build_offer_features(
        self,
        features: pd.DataFrame,
        tables: Dict[str, pd.DataFrame],
        reference_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """Добавляет признаки взаимодействия с офферами."""
        # offer_seens
        if "offer_seens" in tables and len(tables["offer_seens"]) > 0:
            seens = tables["offer_seens"].copy()
            seens_agg = (
                seens.groupby("user_id")
                .agg({"offer_id": "count", "start_date": "first"})
                .reset_index()
            )
            seens_agg.columns = ["user_id", "offers_seen_count", "first_offer_date"]
            features = features.merge(seens_agg, on="user_id", how="left")
            features["offers_seen_count"] = features["offers_seen_count"].fillna(0)

        # offer_activation
        if "offer_activation" in tables and len(tables["offer_activation"]) > 0:
            act = tables["offer_activation"].copy()
            act_agg = (
                act.groupby("user_id")
                .agg({"offer_id": "count", "activation_date": "first"})
                .reset_index()
            )
            act_agg.columns = [
                "user_id",
                "offers_activated_count",
                "first_activation_date",
            ]
            features = features.merge(act_agg, on="user_id", how="left")
            features["offers_activated_count"] = features[
                "offers_activated_count"
            ].fillna(0)

        # offer_reward
        if "offer_reward" in tables and len(tables["offer_reward"]) > 0:
            reward = tables["offer_reward"].copy()
            # Конвертируем reward_amt в numeric
            reward["reward_amt"] = pd.to_numeric(
                reward.get("reward_amt", 0), errors="coerce"
            ).fillna(0)
            reward_agg = (
                reward.groupby("user_id")
                .agg({"offer_id": "count", "reward_amt": "sum"})
                .reset_index()
            )
            reward_agg.columns = ["user_id", "rewards_count", "total_reward_amt"]
            features = features.merge(reward_agg, on="user_id", how="left")
            features["rewards_count"] = (
                features["rewards_count"].fillna(0).astype("int32")
            )
            features["total_reward_amt"] = (
                pd.to_numeric(features["total_reward_amt"], errors="coerce")
                .fillna(0)
                .astype("float32")
            )

        return features

    def _build_interactions(
        self, tables: Dict[str, pd.DataFrame], reference_date: pd.Timestamp
    ) -> pd.DataFrame:
        """
        Строит матрицу взаимодействий user-offer для обучения.
        Positive signal: пользователь активировал оффер или получил reward.
        """
        interactions = []

        # Используем offer_activation как positive signal
        if "offer_activation" in tables and len(tables["offer_activation"]) > 0:
            act = tables["offer_activation"].copy()
            act["activation_date"] = pd.to_datetime(
                act["activation_date"], errors="coerce"
            )

            # Фильтруем активации до reference_date
            act = act[act["activation_date"] <= reference_date]

            act["confidence"] = 1.0
            interactions.append(act[["user_id", "offer_id", "confidence"]])

        # offer_reward как более сильный сигнал
        if "offer_reward" in tables and len(tables["offer_reward"]) > 0:
            reward = tables["offer_reward"].copy()
            reward["event_date"] = pd.to_datetime(reward["event_date"], errors="coerce")
            reward = reward[reward["event_date"] <= reference_date]

            reward["confidence"] = 2.0  # Более сильный сигнал
            interactions.append(reward[["user_id", "offer_id", "confidence"]])

        if len(interactions) == 0:
            return pd.DataFrame(columns=["user_id", "offer_id", "confidence"])

        result = pd.concat(interactions, ignore_index=True)

        # Агрегируем дубликаты (берём макс. confidence)
        result = result.groupby(["user_id", "offer_id"], as_index=False)[
            "confidence"
        ].max()

        # Конвертируем в numeric
        result["user_id"] = (
            pd.to_numeric(result["user_id"], errors="coerce").fillna(0).astype("int64")
        )
        result["offer_id"] = (
            pd.to_numeric(result["offer_id"], errors="coerce").fillna(0).astype("int64")
        )
        result["confidence"] = (
            pd.to_numeric(result["confidence"], errors="coerce")
            .fillna(0)
            .astype("float32")
        )

        return result

    def save_features(
        self,
        features: pd.DataFrame,
        interactions: pd.DataFrame,
        output_path: str = "/app/data",
    ):
        """Сохраняет признаки и взаимодействия."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Конвертируем все числовые колонки в правильные типы
        for col in features.columns:
            if features[col].dtype == "object":
                # Пробуем конвертировать в numeric
                try:
                    features[col] = pd.to_numeric(
                        features[col], errors="coerce"
                    ).fillna(0)
                except:
                    pass  # Оставляем как строку если не конвертируется

        # Конвертируем float64 в float32 для экономии памяти
        float_cols = features.select_dtypes(include=["float64"]).columns
        features[float_cols] = features[float_cols].astype("float32")

        # Конвертируем int64 в int32
        int_cols = features.select_dtypes(include=["int64"]).columns
        features[int_cols] = features[int_cols].astype("int32")

        features.to_parquet(output_path / "features.parquet", index=False)
        interactions.to_parquet(output_path / "interactions.parquet", index=False)

    def load_features(
        self, input_path: str = "/app/data"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Загружает признаки и взаимодействия."""
        input_path = Path(input_path)

        features = pd.read_parquet(input_path / "features.parquet")
        interactions = pd.read_parquet(input_path / "interactions.parquet")

        return features, interactions


def get_feature_builder() -> FeatureBuilder:
    """Возвращает singleton построителя признаков."""
    return FeatureBuilder()
