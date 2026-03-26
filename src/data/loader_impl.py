"""
Модуль загрузки и хранения данных для Look-a-Like сервиса.
Поддерживает загрузку батчами, идемпотентность и хранение в S3.
"""

import os
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
import boto3
from botocore.config import Config
import threading


class DataBuffer:
    """Буфер для временного хранения батчей данных до commit."""

    # Допустимые имена таблиц
    VALID_TABLES = [
        "people", "segments", "transaction", "offer", "merchant",
        "financial_account", "offer_seens", "offer_activation",
        "offer_reward", "receipts"
    ]

    def __init__(self, base_path: str = "/app/data/raw"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _get_batch_hash(self, records: List[Dict]) -> str:
        """Вычисляет хэш батча для идемпотентности."""
        content = json.dumps(records, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()

    def store_batch(
        self,
        version: str,
        table: str,
        batch_id: int,
        total_batches: int,
        records: List[Dict]
    ) -> Tuple[bool, str]:
        """
        Сохраняет батч данных.
        Возвращает (success, message) - было ли сохранение новым или дубликатом.
        """
        # Проверка валидности таблицы
        if table not in self.VALID_TABLES:
            return False, f"Invalid table: {table}"

        with self._lock:
            version_path = self.base_path / version
            version_path.mkdir(parents=True, exist_ok=True)

            # Сохранение батча
            batch_file = version_path / f"{table}_batch_{batch_id}.jsonl"
            meta_file = version_path / f"{table}_batch_{batch_id}.meta"

            # Проверка на идемпотентность
            if batch_file.exists():
                return False, "duplicate"

            # Запись батча
            with open(batch_file, 'w') as f:
                for record in records:
                    f.write(json.dumps(record) + '\n')

            # Сохранение метаданных батча
            with open(meta_file, 'w') as f:
                json.dump({
                    "version": version,
                    "table": table,
                    "batch_id": batch_id,
                    "total_batches": total_batches,
                    "records_count": len(records),
                    "hash": self._get_batch_hash(records)
                }, f)

            return True, "stored"

    def get_version_batches(self, version: str) -> Dict[str, List[int]]:
        """Возвращает список батчей для версии, читая из файлов."""
        version_path = self.base_path / version

        if not version_path.exists():
            return {}

        batches: Dict[str, List[int]] = {}

        # Читаем .meta файлы для определения батчей
        for meta_file in version_path.glob("*_batch_*.meta"):
            try:
                with open(meta_file, 'r') as f:
                    meta = json.load(f)
                table = meta.get('table')
                batch_id = meta.get('batch_id')
                if table and batch_id:
                    if table not in batches:
                        batches[table] = []
                    batches[table].append(int(batch_id))
            except Exception:
                pass

        return batches

    def is_version_complete(
        self,
        version: str,
        expected_tables: Optional[List[str]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Проверяет, все ли таблицы загружены для версии.
        Возвращает (is_complete, list_of_received_tables).
        """
        if expected_tables is None:
            expected_tables = ["people", "transaction", "offer", "merchant"]

        batches = self.get_version_batches(version)
        received_tables = list(batches.keys())

        # Проверяем, что все ожидаемые таблицы присутствуют
        is_complete = all(table in batches for table in expected_tables)
        return is_complete, received_tables

    def commit_version(self, version: str) -> Optional[Path]:
        """
        Фиксирует версию: объединяет батчи в единые файлы.
        Возвращает путь к committed версии или None если ошибка.
        """
        with self._lock:
            version_path = self.base_path / version
            committed_path = version_path / "committed"
            committed_path.mkdir(parents=True, exist_ok=True)

            batches = self.get_version_batches(version)

            if not batches:
                return None

            for table, batch_ids in batches.items():
                # Сортируем батчи по ID
                batch_ids.sort()

                # Объединяем все батчи таблицы
                all_records = []
                for batch_id in batch_ids:
                    batch_file = version_path / f"{table}_batch_{batch_id}.jsonl"
                    if batch_file.exists():
                        with open(batch_file, 'r') as f:
                            for line in f:
                                all_records.append(json.loads(line))

                # Сохраняем объединённую таблицу
                output_file = committed_path / f"{table}.jsonl"
                with open(output_file, 'w') as f:
                    for record in all_records:
                        f.write(json.dumps(record) + '\n')

                # Сохраняем метаданные таблицы
                meta_file = committed_path / f"{table}.meta"
                with open(meta_file, 'w') as f:
                    json.dump({
                        "table": table,
                        "total_records": len(all_records),
                        "batches": batch_ids
                    }, f)

            # Флаг успешного коммита
            (committed_path / ".committed").touch()

            return committed_path

    def is_version_committed(self, version: str) -> bool:
        """Проверяет, была ли версия уже закоммичена."""
        committed_file = self.base_path / version / "committed" / ".committed"
        return committed_file.exists()


class S3Storage:
    """Клиент для работы с S3 (MinIO)."""

    def __init__(
        self,
        endpoint: str = "minio:9000",
        bucket: str = "lookalike-data",
        access_key: str = "minioadmin",
        secret_key: str = "minioadmin"
    ):
        self.endpoint = endpoint
        self.bucket = bucket
        self.client = boto3.client(
            's3',
            endpoint_url=f"http://{endpoint}",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(signature_version='s3v4'),
            region_name='us-east-1'
        )
        self._ensure_bucket()

    def _ensure_bucket(self):
        """Создаёт бакет если не существует."""
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except Exception:
            self.client.create_bucket(Bucket=self.bucket)

    def upload_file(self, local_path: str, s3_key: str):
        """Загружает файл в S3."""
        self.client.upload_file(local_path, self.bucket, s3_key)

    def download_file(self, s3_key: str, local_path: str):
        """Скачивает файл из S3."""
        self.client.download_file(self.bucket, s3_key, local_path)

    def list_files(self, prefix: str = "") -> List[str]:
        """Список файлов в S3."""
        response = self.client.list_objects_v2(
            Bucket=self.bucket,
            Prefix=prefix
        )
        return [obj['Key'] for obj in response.get('Contents', [])]

    def file_exists(self, key: str) -> bool:
        """Проверяет существование файла в S3."""
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False


class DataLoader:
    """Загрузчик данных из различных источников."""

    TABLE_MAPPING = {
        "people": "prod_clients",
        "segments": "prizm_segments",
        "transaction": "prod_financial_transaction",
        "offer": "t_offer",
        "merchant": "t_merchant",
        "financial_account": "financial_account",
        "offer_seens": "offer_seens",
        "offer_activation": "offer_activation",
        "offer_reward": "offer_reward",
        "receipts": "receipts"
    }

    def __init__(self, data_path: Path):
        self.data_path = data_path

    def load_table(self, table: str) -> pd.DataFrame:
        """Загружает таблицу из JSONL файла."""
        # Пробуем загрузить из committed версии
        committed_file = self.data_path / f"{table}.jsonl"

        if not committed_file.exists():
            # Пробуем загрузить из батчей
            batch_files = sorted(self.data_path.glob(f"{table}_batch_*.jsonl"))

            if not batch_files:
                raise FileNotFoundError(f"Table {table} not found in {self.data_path}")

            all_records = []
            for batch_file in batch_files:
                with open(batch_file, 'r') as f:
                    for line in f:
                        all_records.append(json.loads(line))

            return pd.DataFrame(all_records)

        # Загрузка из committed файла
        records = []
        with open(committed_file, 'r') as f:
            for line in f:
                records.append(json.loads(line))

        return pd.DataFrame(records)

    def load_all_tables(self) -> Dict[str, pd.DataFrame]:
        """Загружает все доступные таблицы."""
        tables = {}
        for logical_name in self.TABLE_MAPPING.keys():
            try:
                tables[logical_name] = self.load_table(logical_name)
            except FileNotFoundError:
                pass  # Таблица может отсутствовать
        return tables


def get_buffer() -> DataBuffer:
    """Возвращает singleton буфера данных."""
    return DataBuffer()


def get_s3_storage() -> S3Storage:
    """Возвращает singleton S3 хранилища."""
    return S3Storage(
        endpoint=os.getenv("S3_ENDPOINT", "minio:9000"),
        bucket=os.getenv("S3_BUCKET", "lookalike-data"),
        access_key=os.getenv("S3_ACCESS_KEY", "minioadmin"),
        secret_key=os.getenv("S3_SECRET_KEY", "minioadmin")
    )
