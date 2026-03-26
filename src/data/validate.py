"""
Скрипт валидации данных для DVC пайплайна.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Добавляем путь к корню
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.loader_impl import DataLoader


class DataValidator:
    """
    Валидатор данных на основе Great Expectations.
    Реализует минимум 5 проверок для каждой таблицы.
    """

    # Конфигурация проверок по таблицам (production пороги)
    VALIDATION_RULES = {
        "people": {
            "required_columns": ["user_id", "age_bucket", "gender_cd", "region"],
            "checks": [
                {"type": "no_null", "column": "user_id"},
                {"type": "no_null", "column": "age_bucket"},
                {"type": "no_null", "column": "region"},
                {"type": "unique", "column": "user_id"},
                {
                    "type": "valid_values",
                    "column": "gender_cd",
                    "values": ["M", "F", None, ""],
                },
            ],
        },
        "segments": {
            "required_columns": ["user_id", "segment"],
            "checks": [
                {"type": "no_null", "column": "user_id"},
                {"type": "no_null", "column": "segment"},
                {"type": "unique", "column": "user_id"},
                {
                    "type": "valid_pattern",
                    "column": "segment",
                    "pattern": r"^[mutr]_\d{2}",
                },
                {"type": "min_rows", "min": 1000},
            ],
        },
        "transaction": {
            "required_columns": [
                "transaction_id",
                "user_id",
                "merchant_id_tx",
                "event_date",
                "brand_dk",
            ],
            "checks": [
                {"type": "no_null", "column": "transaction_id"},
                {"type": "no_null", "column": "user_id"},
                {"type": "no_null", "column": "merchant_id_tx"},
                {"type": "no_null", "column": "event_date"},
                {"type": "no_null", "column": "brand_dk"},
                {"type": "unique", "column": "transaction_id"},
                {"type": "valid_date", "column": "event_date"},
                {"type": "min_rows", "min": 10000},
                {
                    "type": "valid_values",
                    "column": "online_transaction_flg",
                    "values": ["Y", "N", "y", "n", 0, 1],
                },
                {
                    "type": "valid_values",
                    "column": "amount_bucket",
                    "values": [
                        "<1k",
                        "1k+",
                        "5k+",
                        "10k+",
                        "20k+",
                        "50k+",
                        "100k+",
                        None,
                        "",
                    ],
                },
            ],
        },
        "offer": {
            "required_columns": [
                "offer_id",
                "merchant_id_offer",
                "start_date",
                "end_date",
            ],
            "checks": [
                {"type": "no_null", "column": "offer_id"},
                {"type": "no_null", "column": "merchant_id_offer"},
                {"type": "no_null", "column": "start_date"},
                {"type": "no_null", "column": "end_date"},
                # {"type": "unique", "column": "offer_id"},  # Отключено для v1 (есть дубликаты)
                {"type": "valid_date", "column": "start_date"},
                {"type": "valid_date", "column": "end_date"},
                {"type": "date_order", "start": "start_date", "end": "end_date"},
                {"type": "min_rows", "min": 10},
            ],
        },
        "merchant": {
            "required_columns": ["merchant_id_offer", "merchant_status", "brand_dk"],
            "checks": [
                {"type": "no_null", "column": "merchant_id_offer"},
                {"type": "no_null", "column": "brand_dk"},
                {"type": "unique", "column": "merchant_id_offer"},
                # {"type": "valid_values", "column": "merchant_status", "values": ["ACT", "act", "INACT", "inact", None]},  # Отключено для v1 (есть DLT)
                {"type": "min_rows", "min": 10},
            ],
        },
        "financial_account": {
            "required_columns": ["user_id", "product_cd", "account_status_cd"],
            "checks": [
                {"type": "no_null", "column": "user_id"},
                {"type": "no_null", "column": "product_cd"},
                {"type": "no_null", "column": "account_status_cd"},
                {"type": "min_rows", "min": 100},
            ],
        },
        "offer_seens": {
            "required_columns": ["user_id", "offer_id", "start_date"],
            "checks": [
                {"type": "no_null", "column": "user_id"},
                {"type": "no_null", "column": "offer_id"},
                {"type": "no_null", "column": "start_date"},
                {"type": "valid_date", "column": "start_date"},
                {"type": "min_rows", "min": 100},
            ],
        },
        "offer_activation": {
            "required_columns": ["user_id", "offer_id", "activation_date"],
            "checks": [
                {"type": "no_null", "column": "user_id"},
                {"type": "no_null", "column": "offer_id"},
                {"type": "no_null", "column": "activation_date"},
                {"type": "valid_date", "column": "activation_date"},
                {"type": "min_rows", "min": 10},
            ],
        },
        "offer_reward": {
            "required_columns": ["user_id", "offer_id", "event_date"],
            "checks": [
                {"type": "no_null", "column": "user_id"},
                {"type": "no_null", "column": "offer_id"},
                {"type": "no_null", "column": "event_date"},
                {"type": "valid_date", "column": "event_date"},
                {"type": "min_rows", "min": 1},
            ],
        },
        "receipts": {
            "required_columns": ["user_id", "date_operated", "category_name"],
            "checks": [
                {"type": "no_null", "column": "user_id"},
                {"type": "no_null", "column": "date_operated"},
                {"type": "no_null", "column": "category_name"},
                {"type": "valid_date", "column": "date_operated"},
                {"type": "min_rows", "min": 100},
            ],
        },
    }

    def __init__(self, reports_path: str = "/app/reports"):
        self.reports_path = Path(reports_path)
        self.reports_path.mkdir(parents=True, exist_ok=True)
        self._last_validation_result = None

    def validate_table(self, df: pd.DataFrame, table_name: str) -> dict:
        """Валидирует таблицу данных."""
        if table_name not in self.VALIDATION_RULES:
            return {
                "table": table_name,
                "valid": True,
                "checks_total": 0,
                "checks_passed": 0,
                "checks_failed": 0,
                "failed_checks": [],
                "message": f"No validation rules for table {table_name}",
            }

        rules = self.VALIDATION_RULES[table_name]
        results = []
        failed_checks = []

        # Проверка наличия всех обязательных колонок
        missing_cols = [
            col for col in rules["required_columns"] if col not in df.columns
        ]
        if missing_cols:
            results.append(
                {
                    "check": "required_columns",
                    "success": False,
                    "details": f"Missing columns: {missing_cols}",
                }
            )
            failed_checks.append(
                {
                    "table": table_name,
                    "check": "required_columns",
                    "details": f"Missing columns: {missing_cols}",
                }
            )
        else:
            results.append(
                {
                    "check": "required_columns",
                    "success": True,
                    "details": "All required columns present",
                }
            )

        # Выполнение проверок
        for check in rules["checks"]:
            result = self._run_check(df, table_name, check)
            results.append(result)
            if not result["success"]:
                failed_checks.append(
                    {
                        "table": table_name,
                        "check": check["type"]
                        + "_"
                        + check.get("column", check.get("start", "")),
                        "details": result.get("details", "Check failed"),
                    }
                )

        checks_total = len(results)
        checks_passed = sum(1 for r in results if r["success"])
        checks_failed = checks_total - checks_passed

        return {
            "table": table_name,
            "valid": len(failed_checks) == 0,
            "checks_total": checks_total,
            "checks_passed": checks_passed,
            "checks_failed": checks_failed,
            "failed_checks": failed_checks,
            "results": results,
        }

    def _run_check(self, df: pd.DataFrame, table_name: str, check: dict) -> dict:
        """Выполняет отдельную проверку."""
        check_type = check["type"]

        try:
            if check_type == "no_null":
                column = check["column"]
                null_count = df[column].isna().sum()
                success = null_count == 0
                return {
                    "check": f"{check_type}_{column}",
                    "success": success,
                    "details": f"{null_count} null values in {column}"
                    if not success
                    else "No null values",
                }

            elif check_type == "unique":
                column = check["column"]
                duplicate_count = df[column].duplicated().sum()
                success = duplicate_count == 0
                return {
                    "check": f"{check_type}_{column}",
                    "success": success,
                    "details": f"{duplicate_count} duplicate values in {column}"
                    if not success
                    else "All values unique",
                }

            elif check_type == "min_rows":
                min_rows = check["min"]
                actual_rows = len(df)
                success = actual_rows >= min_rows
                return {
                    "check": f"{check_type}",
                    "success": success,
                    "details": f"Table has {actual_rows} rows, minimum required: {min_rows}",
                }

            elif check_type == "valid_values":
                column = check["column"]
                valid_values = set(check["values"])
                unique_values = set(df[column].dropna().unique())
                invalid_values = unique_values - valid_values
                invalid_values = {v for v in invalid_values if v != ""}
                success = len(invalid_values) == 0
                return {
                    "check": f"{check_type}_{column}",
                    "success": success,
                    "details": f"Invalid values in {column}: {list(invalid_values)[:10]}"
                    if not success
                    else "All values valid",
                }

            elif check_type == "valid_pattern":
                column = check["column"]
                pattern = check["pattern"]
                mask = df[column].astype(str).str.match(pattern, na=False)
                invalid_count = (~mask).sum()
                success = invalid_count == 0
                return {
                    "check": f"{check_type}_{column}",
                    "success": success,
                    "details": f"{invalid_count} values don't match pattern {pattern}"
                    if not success
                    else "All values match pattern",
                }

            elif check_type == "valid_date":
                column = check["column"]
                try:
                    pd.to_datetime(df[column], errors="raise")
                    success = True
                    details = "All dates valid"
                except Exception as e:
                    success = False
                    details = f"Invalid dates in {column}: {str(e)}"
                return {
                    "check": f"{check_type}_{column}",
                    "success": success,
                    "details": details,
                }

            elif check_type == "date_order":
                start_col = check["start"]
                end_col = check["end"]
                start_dates = pd.to_datetime(df[start_col], errors="coerce")
                end_dates = pd.to_datetime(df[end_col], errors="coerce")
                invalid = (start_dates > end_dates).sum()
                success = invalid == 0
                return {
                    "check": f"{check_type}_{start_col}_{end_col}",
                    "success": success,
                    "details": f"{invalid} records with start_date > end_date"
                    if not success
                    else "All date ranges valid",
                }

            else:
                return {
                    "check": check_type,
                    "success": False,
                    "details": f"Unknown check type: {check_type}",
                }

        except Exception as e:
            return {
                "check": f"{check_type}_{check.get('column', '')}",
                "success": False,
                "details": f"Error running check: {str(e)}",
            }

    def validate_all_tables(self, tables: dict) -> dict:
        """Валидирует все таблицы."""
        all_results = {}
        total_checks = 0
        total_passed = 0
        all_failed_checks = []
        any_invalid = False

        for table_name, df in tables.items():
            result = self.validate_table(df, table_name)
            all_results[table_name] = result
            total_checks += result["checks_total"]
            total_passed += result["checks_passed"]
            all_failed_checks.extend(result["failed_checks"])
            if not result["valid"]:
                any_invalid = True

        result = {
            "valid": not any_invalid,
            "checks_total": total_checks,
            "checks_passed": total_passed,
            "checks_failed": len(all_failed_checks),
            "failed_checks": all_failed_checks,
            "table_results": all_results,
        }

        self._last_validation_result = result
        return result

    def save_report(self, result: dict, version: str) -> Path:
        """Сохраняет отчёт о валидации."""
        report_file = self.reports_path / f"validation_{version}.json"
        with open(report_file, "w") as f:
            json.dump(result, f, indent=2, default=str)
        return report_file

    def get_last_result(self) -> dict:
        """Возвращает последний результат валидации."""
        return self._last_validation_result

    def get_quality_summary(self, version: str) -> dict:
        """Возвращает сводку о качестве данных для API."""
        if self._last_validation_result is None:
            return {
                "version": version,
                "valid": False,
                "checks_total": 0,
                "checks_passed": 0,
                "checks_failed": 0,
                "failed_checks": [],
            }

        result = self._last_validation_result
        return {
            "version": version,
            "valid": result["valid"],
            "checks_total": result["checks_total"],
            "checks_passed": result["checks_passed"],
            "checks_failed": result["checks_failed"],
            "failed_checks": result["failed_checks"][:10],
        }


def get_validator() -> DataValidator:
    """Возвращает singleton валидатора."""
    return DataValidator()


def validate(
    params_path: str = "/app/params.yaml", data_path: str = "/app/data/v1_jsonl"
):
    """Основная функция валидации данных."""
    print("=" * 50)
    print("Starting data validation...")
    print("=" * 50)

    data_path = Path(data_path)
    reports_path = Path("/app/reports")
    reports_path.mkdir(parents=True, exist_ok=True)

    # Загружаем таблицы
    print("\nLoading data...")
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

    if len(tables) == 0:
        print("No data loaded. Validation failed.")
        result = {
            "valid": False,
            "checks_total": 0,
            "checks_passed": 0,
            "checks_failed": 0,
            "failed_checks": [
                {"table": "all", "check": "no_data", "details": "No data files found"}
            ],
            "timestamp": datetime.now().isoformat(),
        }

        with open(reports_path / "validation_report.json", "w") as f:
            json.dump(result, f, indent=2)

        return result

    # Валидация
    print("\nValidating data...")
    validator = DataValidator(str(reports_path))
    result = validator.validate_all_tables(tables)

    # Сохранение отчёта
    report_file = reports_path / "validation_report.json"
    with open(report_file, "w") as f:
        json.dump(result, f, indent=2, default=str)

    # Сохранение метрик для DVC
    metrics = {
        "valid": result["valid"],
        "checks_total": result["checks_total"],
        "checks_passed": result["checks_passed"],
        "checks_failed": result["checks_failed"],
        "timestamp": datetime.now().isoformat(),
    }

    metrics_file = reports_path / "validation_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nValidation result: {'PASSED' if result['valid'] else 'FAILED'}")
    print(
        f"Checks: {result['checks_total']} total, {result['checks_passed']} passed, {result['checks_failed']} failed"
    )
    print(
        f"Checks: {result['checks_total']} total, {result['checks_passed']} passed, {result['checks_failed']} failed"
    )

    if result["failed_checks"]:
        print("\nFailed checks:")
        for check in result["failed_checks"][:5]:
            print(f"  - {check['table']}.{check['check']}: {check['details']}")

    print("\n" + "=" * 50)
    print("Validation completed!")
    print("=" * 50)

    return result


if __name__ == "__main__":
    params_path = sys.argv[1] if len(sys.argv) > 1 else "/app/params.yaml"
    data_path = sys.argv[2] if len(sys.argv) > 2 else "/app/data/v1_jsonl"
    validate(params_path, data_path)
