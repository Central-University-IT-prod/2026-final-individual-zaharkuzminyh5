"""
Модуль с реализацией ALS модели для look-a-like.
"""

import logging
from typing import List, Tuple

import implicit
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LookalikeModel:
    """
    Обертка для ALS модели из библиотеки implicit.
    """

    def __init__(
        self,
        factors: int = 128,
        iterations: int = 15,
        regularization: float = 0.01,
        random_state: int = 42,
    ):
        """
        Инициализация модели.

        Args:
            factors: Размерность латентных векторов (embeddings).
            iterations: Количество итераций ALS.
            regularization: Коэффициент регуляризации.
            random_state: Для воспроизводимости.
        """
        self.model = implicit.als.AlternatingLeastSquares(
            factors=factors,
            iterations=iterations,
            regularization=regularization,
            random_state=random_state,
        )
        self.user_map = None
        self.item_map = None
        self.interaction_matrix = None
        self.is_trained = False
        self.user_features = None
        self.train_metrics = None

    def _create_maps(self, users: np.ndarray, items: np.ndarray):
        """Создает маппинги ID пользователей и айтемов в индексы матрицы."""
        self.user_map = {user_id: i for i, user_id in enumerate(users)}
        self.item_map = {item_id: i for i, item_id in enumerate(items)}
        self.reverse_user_map = {i: u for u, i in self.user_map.items()}
        self.reverse_item_map = {i: item for item, i in self.item_map.items()}

    def _create_sparse_matrix(self, interactions: pd.DataFrame) -> csr_matrix:
        """
        Создает разреженную матрицу user-item из взаимодействий.
        """
        users = interactions["user_id"].unique()
        items = interactions["offer_id"].unique()

        self._create_maps(users, items)

        user_ids = interactions["user_id"].map(self.user_map).values
        item_ids = interactions["offer_id"].map(self.item_map).values
        confidence = interactions["confidence"].values

        return coo_matrix(
            (confidence, (user_ids, item_ids)),
            shape=(len(self.user_map), len(self.item_map)),
        ).tocsr()

    def fit(self, interactions: pd.DataFrame, features: pd.DataFrame):
        """
        Обучает модель на данных о взаимодействиях.

        Args:
            interactions: DataFrame с колонками [user_id, offer_id, confidence].
            features: DataFrame с признаками пользователей.
        """
        logger.info("Creating sparse matrix for training...")
        self.interaction_matrix = self._create_sparse_matrix(interactions)
        self.user_features = features

        logger.info(
            f"Training ALS model on matrix with shape {self.interaction_matrix.shape}..."
        )
        self.model.fit(self.interaction_matrix)
        self.is_trained = True
        logger.info("Model training completed.")

        self.train_metrics = {
            "num_users": len(self.user_map),
            "num_items": len(self.item_map),
            "model_factors": self.model.factors,
            "model_iterations": self.model.iterations,
        }
        return self.train_metrics

    def predict(
        self, seed_users: List[int], top_n: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Находит похожих пользователей.

        Args:
            seed_users: Список ID пользователей для поиска похожих.
            top_n: Количество похожих пользователей для возврата.

        Returns:
            (users, scores): Массивы с ID пользователей и их similarity score.
        """
        if self.user_map is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        valid_seed_indices = [
            self.user_map[u] for u in seed_users if u in self.user_map
        ]

        if not valid_seed_indices:
            logger.warning(
                "No seed users found in the model. Returning empty prediction."
            )
            return np.array([]), np.array([])

        user_indices, scores = self.model.similar_items(
            valid_seed_indices, N=top_n + len(seed_users)
        )

        df = pd.DataFrame(
            {"user_idx": user_indices.flatten(), "score": scores.flatten()}
        )

        agg_scores = df.groupby("user_idx")["score"].mean().sort_values(ascending=False)

        seed_indices_set = set(valid_seed_indices)
        agg_scores = agg_scores[~agg_scores.index.isin(seed_indices_set)]

        top_users_indices = agg_scores.head(top_n).index.values
        top_scores = agg_scores.head(top_n).values

        top_user_ids = np.array([self.reverse_user_map[i] for i in top_users_indices])

        return top_user_ids, top_scores

    def recommend(
        self, user_id: int, top_n: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Рекомендует офферы для одного пользователя.
        """
        if self.user_map is None or user_id not in self.user_map:
            logger.warning(f"User {user_id} not found in the model.")
            return np.array([]), np.array([])

        user_idx = self.user_map[user_id]

        item_indices, scores = self.model.recommend(
            user_idx,
            self.interaction_matrix[user_idx],
            N=top_n,
            filter_already_liked_items=True,
        )

        top_offer_ids = np.array([self.reverse_item_map[i] for i in item_indices])

        return top_offer_ids, scores
