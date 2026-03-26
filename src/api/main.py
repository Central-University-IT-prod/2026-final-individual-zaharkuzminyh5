"""
REST API для Look-a-Like сервиса.
Реализует все эндпоинты согласно openapi.yml.
"""

import asyncio
import json
import logging
import os
import pickle
import sys
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import uvicorn
import yaml
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader_impl import DataBuffer, DataLoader, get_buffer, get_s3_storage
from data.validate import DataValidator, get_validator
from ml.features import FeatureBuilder, get_feature_builder
from ml.model import LookalikeModel
from monitoring.drift import DriftDetector, get_drift_detector

# ============================================================================
# Pydantic модели для API
# ============================================================================


class DataBatchRequest(BaseModel):
    version: str


class DataBatchRequest(BaseModel):
    version: str
    table: str
    batch_id: int = Field(ge=1)
    total_batches: int = Field(ge=1)
    records: List[Dict[str, Any]]

    @validator("table")
    def validate_table(cls, v):
        allowed_tables = [
            "people",
            "segments",
            "transaction",
            "offer",
            "merchant",
            "financial_account",
            "offer_seens",
            "offer_activation",
            "offer_reward",
            "receipts",
        ]
        if v not in allowed_tables:
            raise ValueError(f"Unknown table: {v}. Allowed: {allowed_tables}")
        return v


class DataBatchResponse(BaseModel):
    status: str = "accepted"
    table: str
    batch_id: int


class DataCommitRequest(BaseModel):
    version: str


class DataCommitResponse(BaseModel):
    status: str = "accepted"
    tables_received: List[str]


class StatusResponse(BaseModel):
    ready: bool
    model_version: str
    data_version: str
    pipeline_status: str


class LookalikeRequest(BaseModel):
    merchant_id: int = Field(ge=1)
    offer_id: int = Field(ge=1)
    top_n: int = Field(ge=1, le=1000)


class LookalikeResponse(BaseModel):
    merchant_id: int
    offer_id: int
    audience: List[Dict[str, Any]]
    audience_size: int
    model_version: str
    reasons: List[Dict[str, Any]]


class LookalikeBatchRequest(BaseModel):
    requests: List[LookalikeRequest]


class LookalikeBatchResponse(BaseModel):
    results: List[LookalikeResponse]


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    trained_on: str
    features_count: Optional[int] = None
    train_metrics: Optional[Dict[str, float]] = None


class DriftResponse(BaseModel):
    drift_detected: bool
    drift_score: float
    action_taken: str


class DataQualityResponse(BaseModel):
    version: str
    valid: bool
    checks_total: int
    checks_passed: int
    checks_failed: int
    failed_checks: List[Dict[str, str]]


class ExperimentRun(BaseModel):
    run_id: str
    data_version: str
    model_version: str
    metrics: Dict[str, float]
    params: Optional[Dict[str, str]] = None
    timestamp: str


class ExperimentsResponse(BaseModel):
    experiments: List[ExperimentRun]


# ============================================================================
# Глобальное состояние сервиса
# ============================================================================


class ServiceState:
    """Глобальное состояние сервиса."""

    def __init__(self, state_path: str = "/app/data/state.json"):
        self.buffer: Optional[DataBuffer] = None
        self.validator: Optional[DataValidator] = None
        self.model: Optional[LookalikeModel] = None
        self.drift_detector: Optional[DriftDetector] = None

        self.current_data_version: str = "none"
        self.current_model_version: str = "0.0"
        self.trained_on_version: str = "none"

        self.pipeline_status: str = "idle"  # idle, running, failed
        self.pipeline_lock: threading.Lock = threading.Lock()

        self.last_validation_result: Optional[Dict] = None
        self.last_drift_result: Optional[Dict] = None

        self.committed_versions: set = set()  # Уже закоммиченные версии
        self.experiments: List[Dict] = []  # История экспериментов

        self.offers_set: set = set()  # Известные offer_id
        self.merchants_set: set = set()  # Известные merchant_id

        self.state_path = Path(state_path)
        self.load_state()

    def load_state(self):
        """Загружает состояние из файла."""
        if self.state_path.exists():
            try:
                with open(self.state_path, "r") as f:
                    data = json.load(f)
                self.committed_versions = set(data.get("committed_versions", []))
                self.offers_set = set(data.get("offers", []))
                self.merchants_set = set(data.get("merchants", []))
                self.current_model_version = data.get("model_version", "0.0")
                self.trained_on_version = data.get("trained_on", "none")
                self.current_data_version = data.get("data_version", "none")
                logger.info(f"Loaded state from {self.state_path}")
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")

    def save_state(self):
        """Сохраняет состояние в файл."""
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_path, "w") as f:
                # Конвертируем numpy типы в Python типы
                def convert(obj):
                    import numpy as np

                    if isinstance(obj, (set, frozenset)):
                        return [int(x) if hasattr(x, "item") else x for x in obj]
                    elif isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, dict):
                        return {str(k): convert(v) for k, v in obj.items()}
                    elif isinstance(obj, (list, tuple)):
                        return [convert(i) for i in obj]
                    return obj

                json.dump(
                    {
                        "committed_versions": convert(self.committed_versions),
                        "offers": convert(self.offers_set),
                        "merchants": convert(self.merchants_set),
                        "model_version": self.current_model_version,
                        "trained_on": self.trained_on_version,
                        "data_version": self.current_data_version,
                    },
                    f,
                    default=str,
                )
            logger.debug("State saved")
        except Exception as e:
            logger.warning(f"Failed to save state: {e}")

    def reset_for_new_version(self, version: str):
        """Сбрасывает состояние для новой версии."""
        self.current_data_version = version
        self.pipeline_status = "running"

    def set_model_trained(self, version: str, model_version: str, metrics: Dict):
        """Обновляет состояние после обучения модели."""
        self.trained_on_version = version
        self.current_model_version = model_version
        self.pipeline_status = "idle"

        # Добавляем эксперимент
        self.experiments.append(
            {
                "run_id": f"run_{len(self.experiments) + 1}",
                "data_version": version,
                "model_version": model_version,
                "metrics": metrics,
                "timestamp": datetime.now().isoformat(),
            }
        )

        self.save_state()

    def set_pipeline_failed(self):
        """Отмечает пайплайн как неудачный."""
        self.pipeline_status = "failed"

    def set_pipeline_idle(self):
        """Возвращает пайплайн в idle."""
        self.pipeline_status = "idle"


state = ServiceState()


# ============================================================================
# Lifecycle
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Инициализация и shutdown приложения."""
    # Startup
    logger.info("Initializing service...")

    state.buffer = get_buffer()
    state.validator = get_validator()
    state.drift_detector = get_drift_detector()

    # Пробуем загрузить существующую модель
    model_path = Path("/app/models/model.pkl")
    if model_path.exists():
        try:
            with open(model_path, "rb") as f:
                state.model = pickle.load(f)
            state.current_model_version = "1.0"  # Or load from metadata
            logger.info(f"Loaded existing model from {model_path}")
        except Exception as e:
            logger.warning(f"Failed to load model: {e}")
            state.model = None

    logger.info("Service initialized!")

    yield

    # Shutdown
    logger.info("Shutting down service...")


# ============================================================================
# FastAPI приложение
# ============================================================================

app = FastAPI(
    title="Look-a-Like Service",
    description="ML service for building look-a-like audiences",
    version="3.0",
    lifespan=lifespan,
)


# ============================================================================
# Health эндпоинты
# ============================================================================


@app.get("/ready")
async def ready():
    """Проверка готовности сервиса."""
    return {"status": "ok"}


@app.get("/status")
async def get_status():
    """Текущее состояние сервиса."""
    return {
        "ready": state.model is not None and state.model.is_trained,
        "model_version": state.current_model_version,
        "data_version": state.current_data_version,
        "pipeline_status": state.pipeline_status,
    }


# ============================================================================
# Data эндпоинты
# ============================================================================


@app.post("/data/batch", response_model=DataBatchResponse)
async def data_batch(request: DataBatchRequest):
    """Загрузка батча данных."""
    if state.buffer is None:
        raise HTTPException(status_code=500, detail="Service not initialized")

    # Проверяем обязательные поля
    if not request.records and request.records != []:
        raise HTTPException(status_code=400, detail="Missing records field")

    # Сохраняем батч
    success, message = state.buffer.store_batch(
        version=request.version,
        table=request.table,
        batch_id=request.batch_id,
        total_batches=request.total_batches,
        records=request.records,
    )

    if message == "duplicate":
        # Идемпотентность - возвращаем успех, но не дублируем
        return DataBatchResponse(
            status="accepted", table=request.table, batch_id=request.batch_id
        )

    return DataBatchResponse(
        status="accepted", table=request.table, batch_id=request.batch_id
    )


@app.post("/data/commit", response_model=DataCommitResponse)
async def data_commit(request: DataCommitRequest, background_tasks: BackgroundTasks):
    """Commit версии данных и запуск пайплайна."""
    if state.buffer is None:
        raise HTTPException(status_code=500, detail="Service not initialized")

    version = request.version

    # Проверка на идемпотентность - если версия уже закоммичена
    if version in state.committed_versions:
        # Возвращаем список таблиц которые были загружены для этой версии
        _, received_tables = state.buffer.is_version_complete(
            version=version,
            expected_tables=["people", "transaction", "offer", "merchant"],
        )
        return DataCommitResponse(status="accepted", tables_received=received_tables)

    # Проверяем наличие данных
    is_complete, received_tables = state.buffer.is_version_complete(
        version=version, expected_tables=["people", "transaction", "offer", "merchant"]
    )

    if not is_complete:
        # Неполные данные - пропускаем обучение
        state.committed_versions.add(version)
        state.current_data_version = version
        state.last_validation_result = {
            "valid": False,
            "checks_total": 1,
            "checks_passed": 0,
            "checks_failed": 1,
            "failed_checks": [
                {
                    "table": "all",
                    "check": "incomplete_data",
                    "details": f"Missing tables: {[t for t in ['people', 'transaction', 'offer', 'merchant'] if t not in received_tables]}",
                }
            ],
        }
        state.save_state()

        return DataCommitResponse(status="accepted", tables_received=received_tables)

    # Фиксируем версию
    state.committed_versions.add(version)
    state.reset_for_new_version(version)
    state.save_state()

    # Запускаем пайплайн в отдельном потоке (не блокировать event loop)
    thread = threading.Thread(target=run_pipeline, args=(version,), daemon=True)
    thread.start()

    return DataCommitResponse(status="accepted", tables_received=received_tables)


def build_merchant_offer_maps_local(tables: dict) -> tuple:
    """Строит маппинги merchant-offer."""
    merchant_offer_map = {}
    offer_merchant_map = {}

    if "offer" in tables:
        for _, row in tables["offer"].iterrows():
            merchant_id = row.get("merchant_id_offer")
            offer_id = row.get("offer_id")

            if pd.notna(merchant_id) and pd.notna(offer_id):
                try:
                    merchant_id = int(float(merchant_id))
                    offer_id = int(float(offer_id))

                    if merchant_id not in merchant_offer_map:
                        merchant_offer_map[merchant_id] = []
                    merchant_offer_map[merchant_id].append(offer_id)
                    offer_merchant_map[offer_id] = merchant_id
                except (ValueError, TypeError):
                    pass

    return merchant_offer_map, offer_merchant_map


def build_existing_customers_local(tables: dict, reference_date: str) -> dict:
    """
    Строит маппинг merchant -> existing customers.
    """
    existing_customers = {}

    if "transaction" not in tables or "merchant" not in tables:
        return existing_customers

    tx = tables["transaction"].copy()
    merchant = tables["merchant"].copy()

    # Связываем транзакции с мерчантами через brand_dk
    if "brand_dk" in tx.columns and "brand_dk" in merchant.columns:
        # Конвертируем в numeric
        merchant["brand_dk"] = pd.to_numeric(
            merchant["brand_dk"], errors="coerce"
        ).fillna(0)
        merchant["merchant_id_offer"] = pd.to_numeric(
            merchant["merchant_id_offer"], errors="coerce"
        ).fillna(0)

        brand_merchant_map = dict(
            zip(merchant["brand_dk"], merchant["merchant_id_offer"])
        )

        tx["brand_dk"] = pd.to_numeric(tx["brand_dk"], errors="coerce").fillna(0)
        tx["merchant_id"] = tx["brand_dk"].map(brand_merchant_map)

        # Для каждого мерчанта находим клиентов
        for merchant_id in tx["merchant_id"].dropna().unique():
            if merchant_id > 0:
                customers = set(
                    tx[tx["merchant_id"] == merchant_id]["user_id"].unique()
                )
                existing_customers[int(merchant_id)] = customers

    return existing_customers


def run_pipeline(version: str):
    """
    Запускает пайплайн обработки данных.
    Выполняется в фоне.
    """
    print(f"\n{'=' * 50}")
    print(f"Running pipeline for version {version}")
    print(f"{'=' * 50}")

    try:
        # Commit версии - объединяем батчи
        print("\nCommitting version...")
        committed_path = state.buffer.commit_version(version)
        print(f"Committed to: {committed_path}")

        # Загружаем таблицы
        print("\nLoading tables...")
        loader = DataLoader(committed_path)
        tables = loader.load_all_tables()

        if len(tables) == 0:
            print("No data loaded!")
            state.last_validation_result = {
                "valid": False,
                "checks_total": 1,
                "checks_passed": 0,
                "checks_failed": 1,
                "failed_checks": [
                    {"table": "all", "check": "no_data", "details": "No data loaded"}
                ],
            }
            state.set_pipeline_failed()
            return

        # Обновляем известные офферы и мерчанты
        if "offer" in tables:
            state.offers_set.update(tables["offer"]["offer_id"].unique())
        if "merchant" in tables:
            state.merchants_set.update(tables["merchant"]["merchant_id_offer"].unique())
        state.save_state()

        # Валидация
        print("\nValidating data...")
        validation_result = state.validator.validate_all_tables(tables)
        state.last_validation_result = validation_result

        # Сохраняем отчёт
        state.validator.save_report(validation_result, version)

        if not validation_result["valid"]:
            print(
                f"Validation failed: {validation_result['checks_failed']} checks failed"
            )
            state.set_pipeline_idle()
            return

        print("Validation passed!")

        # Построение признаков
        print("\nBuilding features...")
        feature_builder = get_feature_builder()
        reference_date = "2026-02-18"

        if "transaction" in tables and len(tables["transaction"]) > 0:
            reference_date = tables["transaction"]["event_date"].max()

        features, interactions = feature_builder.build_features(tables, reference_date)
        feature_builder.save_features(features, interactions, str(committed_path))

        print(f"Features shape: {features.shape}")

        # Проверка дрейфа
        print("\nChecking drift...")

        # Загружаем референсные признаки если нет
        if state.drift_detector._reference_features is None:
            ref_path = Path("/app/data/reference_features.parquet")
            if ref_path.exists():
                state.drift_detector.load_reference(str(ref_path))
            else:
                # Сохраняем текущие как референсные
                state.drift_detector.save_reference(features, str(ref_path))
                print("Saved reference features")

        # Детектируем дрейф
        drift_result = state.drift_detector.detect_drift(features)
        state.last_drift_result = drift_result

        print(
            f"Drift detected: {drift_result['drift_detected']}, score: {drift_result['drift_score']}"
        )

        # Решение о переобучении
        # Обучаем если: дрейф обнаружен ИЛИ это первая версия (model_version = 0.0)
        should_retrain = drift_result["drift_detected"] or (
            state.current_model_version == "0.0"
        )

        if should_retrain:
            print("\nRetraining model...")

            # Строим маппинги
            from ml.train import load_tables

            merchant_offer_map, offer_merchant_map = build_merchant_offer_maps_local(
                tables
            )
            existing_customers = build_existing_customers_local(tables, reference_date)

            # Обучаем модель
            model = LookalikeModel()
            train_metrics = model.fit(
                features=features,
                interactions=interactions,
                merchant_offer_map=merchant_offer_map,
                offer_merchant_map=offer_merchant_map,
                existing_customers=existing_customers,
            )

            # Сохраняем модель
            model_dir = Path("/app/models")
            model_dir.mkdir(exist_ok=True)
            model_path = model_dir / "model.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            state.model = model

            # Обновляем состояние
            new_model_version = (
                f"{int(state.current_model_version.split('.')[0]) + 1}.0"
            )
            state.set_model_trained(version, new_model_version, train_metrics)

            # MLflow логирование
            try:
                import mlflow

                mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
                mlflow.set_tracking_uri(mlflow_uri)

                with mlflow.start_run(run_name=f"lookalike-{version}"):
                    # Параметры
                    mlflow.log_param("data_version", version)
                    mlflow.log_param("model_type", "als")
                    mlflow.log_param("n_factors", 64)

                    # Метрики
                    for metric_name, value in train_metrics.items():
                        mlflow.log_metric(metric_name, value)

                    # Артефакты
                    mlflow.log_artifact(str(model_path))

                    logger.info(f"MLflow run logged: {mlflow.active_run().info.run_id}")
            except Exception as e:
                logger.warning(f"Failed to log to MLflow: {e}")

            # Сохраняем референсные признаки для следующей проверки
            state.drift_detector.save_reference(
                features, str(Path("/app/data/reference_features.parquet"))
            )

            print(f"Model retrained! New version: {new_model_version}")
            print(f"Train metrics: {train_metrics}")
        else:
            print("\nNo drift detected, skipping retraining")
            state.set_pipeline_idle()

        logger.info(f"\n{'=' * 50}")
        logger.info(f"Pipeline completed for version {version}")
        logger.info(f"{'=' * 50}\n")

    except Exception as e:
        logger.error(f"Pipeline failed for version {version}: {e}")
        logger.exception("Traceback:")
        state.set_pipeline_failed()


# ============================================================================
# Inference эндпоинты
# ============================================================================


@app.post("/lookalike", response_model=LookalikeResponse)
async def lookalike(request: LookalikeRequest):
    """Предсказание look-a-like аудитории."""
    if state.model is None or not state.model.is_trained:
        raise HTTPException(status_code=503, detail="Model not ready")

    offer_id = request.offer_id
    merchant_id = request.merchant_id
    top_n = request.top_n

    # Проверяем наличие оффера
    if offer_id not in state.offers_set:
        raise HTTPException(status_code=404, detail=f"Offer {offer_id} not found")

    # Получаем предсказания
    try:
        audience = state.model.predict(merchant_id, offer_id, top_n)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    # Получаем reasons для первого пользователя (для примера)
    reasons = [{"feature": "collaborative_filtering", "impact": 0.5}]
    if audience and state.model.user_features is not None:
        first_user = audience[0]["user_id"]
        reasons = state.model.get_reasons(first_user, merchant_id, offer_id)

    return LookalikeResponse(
        merchant_id=merchant_id,
        offer_id=offer_id,
        audience=audience,
        audience_size=len(audience),
        model_version=state.current_model_version,
        reasons=reasons,
    )


@app.post("/lookalike/batch", response_model=LookalikeBatchResponse)
async def lookalike_batch(request: LookalikeBatchRequest):
    """Пакетное предсказание."""
    results = []

    for req in request.requests:
        try:
            response = await lookalike(req)
            results.append(response)
        except HTTPException as e:
            # Возвращаем ошибку в результате
            results.append(
                {
                    "merchant_id": req.merchant_id,
                    "offer_id": req.offer_id,
                    "audience": [],
                    "audience_size": 0,
                    "model_version": state.current_model_version,
                    "reasons": [],
                    "error": str(e.detail),
                }
            )

    return LookalikeBatchResponse(results=results)


# ============================================================================
# Monitoring эндпоинты
# ============================================================================


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """Информация о модели."""
    if state.model is None:
        raise HTTPException(status_code=404, detail="Model not found")

    return ModelInfoResponse(
        model_name="lookalike-cf",
        model_version=state.current_model_version,
        trained_on=state.trained_on_version,
        features_count=len(state.model.user_features.columns) - 1
        if state.model.user_features is not None
        else 0,
        train_metrics=state.model.train_metrics,
    )


@app.get("/monitoring/drift", response_model=DriftResponse)
async def monitoring_drift():
    """Статус дрейфа."""
    if state.last_drift_result is None:
        return DriftResponse(drift_detected=False, drift_score=0.0, action_taken="none")

    result = state.last_drift_result
    # Определяем action_taken
    if state.last_validation_result and not state.last_validation_result.get(
        "valid", True
    ):
        action = "skipped"  # Валидация не прошла
    elif result["drift_detected"]:
        action = "retrained"
    else:
        action = "none"

    return DriftResponse(
        drift_detected=result["drift_detected"],
        drift_score=result["drift_score"],
        action_taken=action,
    )


@app.get("/monitoring/data-quality", response_model=DataQualityResponse)
async def monitoring_data_quality():
    """Качество данных."""
    if state.last_validation_result is None:
        return DataQualityResponse(
            version=state.current_data_version,
            valid=False,
            checks_total=0,
            checks_passed=0,
            checks_failed=0,
            failed_checks=[],
        )

    result = state.last_validation_result
    return DataQualityResponse(
        version=state.current_data_version,
        valid=result["valid"],
        checks_total=result["checks_total"],
        checks_passed=result["checks_passed"],
        checks_failed=result["checks_failed"],
        failed_checks=result.get("failed_checks", [])[:10],
    )


@app.get("/experiments", response_model=ExperimentsResponse)
async def get_experiments():
    """История экспериментов."""
    # Пробуем загрузить из MLflow
    try:
        import mlflow

        # MLflow URI: внутри docker используем mlflow:5000, снаружи localhost:5000
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(mlflow_uri)
        logger.info(f"MLflow tracking URI: {mlflow_uri}")

        experiments = []
        for run in mlflow.search_runs(experiment_names=["lookalike-cf"]).itertuples():
            experiments.append(
                {
                    "run_id": run.run_id,
                    "data_version": getattr(
                        run, "params_data_version", state.current_data_version
                    ),
                    "model_version": f"v{run.Index + 1}.0"
                    if hasattr(run, "Index")
                    else "1.0",
                    "metrics": {
                        k.replace("metrics_", ""): v
                        for k, v in run._asdict().items()
                        if k.startswith("metrics_") and not pd.isna(v)
                    },
                    "timestamp": run.start_time.isoformat()
                    if hasattr(run, "start_time")
                    else datetime.now().isoformat(),
                }
            )

        if experiments:
            return ExperimentsResponse(experiments=experiments)
    except Exception as e:
        logger.warning(f"Failed to load experiments from MLflow: {e}")

    # Возвращаем локальные эксперименты
    return ExperimentsResponse(experiments=state.experiments)


# ============================================================================
# Запуск
# ============================================================================

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=80, reload=False)
