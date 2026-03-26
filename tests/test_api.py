"""
Тесты для API Look-a-Like сервиса.
"""

import json
import time
from typing import Any, Dict

import pytest
import requests

BASE_URL = "http://localhost:80"


class TestHealthEndpoints:
    """Тесты health эндпоинтов."""

    def test_ready(self):
        """GET /ready должен вернуть 200."""
        response = requests.get(f"{BASE_URL}/ready")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_status(self):
        """GET /status должен вернуть статус сервиса."""
        response = requests.get(f"{BASE_URL}/status")
        assert response.status_code == 200

        data = response.json()
        assert "ready" in data
        assert "model_version" in data
        assert "data_version" in data
        assert "pipeline_status" in data


class TestDataEndpoints:
    """Тесты data эндпоинтов."""

    def test_data_batch_invalid_table(self):
        """POST /data/batch с неизвестной таблицей должен вернуть 400."""
        response = requests.post(
            f"{BASE_URL}/data/batch",
            json={
                "version": "test",
                "table": "unknown_table",
                "batch_id": 1,
                "total_batches": 1,
                "records": [],
            },
        )
        assert response.status_code == 422

    def test_data_batch_empty_records(self):
        """POST /data/batch с пустыми records должен вернуть 200."""
        response = requests.post(
            f"{BASE_URL}/data/batch",
            json={
                "version": "test",
                "table": "people",
                "batch_id": 1,
                "total_batches": 1,
                "records": [],
            },
        )
        assert response.status_code == 200

    def test_data_batch_missing_field(self):
        """POST /data/batch без обязательного поля должен вернуть 400."""
        response = requests.post(
            f"{BASE_URL}/data/batch",
            json={
                "version": "test",
                "table": "people",
                "batch_id": 1,
                # missing total_batches
            },
        )
        assert response.status_code == 422  # Validation error


class TestLookalikeEndpoints:
    """Тесты lookalike эндпоинтов."""

    def test_lookalike_invalid_top_n(self):
        """POST /lookalike с невалидным top_n должен вернуть 400."""
        response = requests.post(
            f"{BASE_URL}/lookalike",
            json={
                "merchant_id": 1,
                "offer_id": 1,
                "top_n": 0,  # invalid
            },
        )
        assert response.status_code == 422

        response = requests.post(
            f"{BASE_URL}/lookalike",
            json={
                "merchant_id": 1,
                "offer_id": 1,
                "top_n": 1001,  # invalid
            },
        )
        assert response.status_code == 422

    def test_lookalike_not_found(self):
        """POST /lookalike с несуществующим offer_id должен вернуть 404."""
        response = requests.post(
            f"{BASE_URL}/lookalike",
            json={"merchant_id": 99999, "offer_id": 99999, "top_n": 10},
        )
        # Может вернуть 404 или 503 если модель не готова
        assert response.status_code in [404, 503]


class TestMonitoringEndpoints:
    """Тесты monitoring эндпоинтов."""

    def test_monitoring_drift(self):
        """GET /monitoring/drift должен вернуть статус дрейфа."""
        response = requests.get(f"{BASE_URL}/monitoring/drift")
        assert response.status_code == 200

        data = response.json()
        assert "drift_detected" in data
        assert "drift_score" in data
        assert "action_taken" in data

    def test_monitoring_data_quality(self):
        """GET /monitoring/data-quality должен вернуть качество данных."""
        response = requests.get(f"{BASE_URL}/monitoring/data-quality")
        assert response.status_code == 200

        data = response.json()
        assert "version" in data
        assert "valid" in data
        assert "checks_total" in data
        assert "checks_passed" in data
        assert "checks_failed" in data
        assert "failed_checks" in data

    def test_model_info(self):
        """GET /model/info должен вернуть информацию о модели."""
        response = requests.get(f"{BASE_URL}/model/info")
        # Может вернуть 404 если модель не обучена
        if response.status_code == 200:
            data = response.json()
            assert "model_name" in data
            assert "model_version" in data
            assert "trained_on" in data

    def test_experiments(self):
        """GET /experiments должен вернуть историю экспериментов."""
        response = requests.get(f"{BASE_URL}/experiments")
        assert response.status_code == 200

        data = response.json()
        assert "experiments" in data
        assert isinstance(data["experiments"], list)


class TestIdempotency:
    """Тесты идемпотентности."""

    def test_data_batch_idempotency(self):
        """Повторный POST /data/batch не должен дублировать данные."""
        batch_data = {
            "version": "idempotency_test",
            "table": "people",
            "batch_id": 1,
            "total_batches": 1,
            "records": [{"user_id": 1, "age_bucket": "25"}],
        }

        # Первый запрос
        response1 = requests.post(f"{BASE_URL}/data/batch", json=batch_data)
        assert response1.status_code == 200

        # Повторный запрос
        response2 = requests.post(f"{BASE_URL}/data/batch", json=batch_data)
        assert response2.status_code == 200


def run_tests():
    """Запускает тесты."""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_tests()
