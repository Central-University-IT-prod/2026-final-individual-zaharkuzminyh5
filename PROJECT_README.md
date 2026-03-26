# Look-a-Like ML Service

ML-сервис для построения look-a-like аудитории под конкретный оффер партнёра банка.

## Структура проекта

```
.
├── src/
│   ├── api/           # REST API (FastAPI)
│   │   └── main.py
│   ├── data/          # Загрузка и валидация данных
│   │   ├── loader.py
│   │   └── validate.py
│   ├── ml/            # ML модель и фичи
│   │   ├── model.py
│   │   ├── features.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── build_features.py
│   └── monitoring/    # Мониторинг дрейфа
│       └── drift.py
├── tests/
│   ├── test_api.py    # API тесты
│   └── test_local.py  # Локальное тестирование
├── reports/           # Отчёты DVC
├── data/              # Данные
├── models/            # Модели
├── great_expectations/ # GE конфигурация
├── Dockerfile
├── docker-compose.yml
├── dvc.yaml
├── params.yaml
└── requirements.txt
```

## Быстрый старт

### 1. Запуск через Docker Compose

```bash
docker compose up -d
```

Сервис будет доступен на http://localhost:80

### 2. Проверка готовности

```bash
curl http://localhost:80/ready
```

### 3. Загрузка данных

```bash
# Загрузка батча
curl -X POST http://localhost:80/data/batch \
  -H "Content-Type: application/json" \
  -d '{
    "version": "v1",
    "table": "people",
    "batch_id": 1,
    "total_batches": 1,
    "records": [{"user_id": 1, "age_bucket": "25", "gender_cd": "M", "region": "Москва"}]
  }'

# Commit версии
curl -X POST http://localhost:80/data/commit \
  -H "Content-Type: application/json" \
  -d '{"version": "v1"}'
```

### 4. Предсказание

```bash
curl -X POST http://localhost:80/lookalike \
  -H "Content-Type: application/json" \
  -d '{
    "merchant_id": 75,
    "offer_id": 42,
    "top_n": 100
  }'
```

## API Endpoints

| Endpoint | Method | Описание |
|----------|--------|----------|
| `/ready` | GET | Проверка готовности |
| `/status` | GET | Статус сервиса |
| `/data/batch` | POST | Загрузка батча данных |
| `/data/commit` | POST | Commit версии данных |
| `/lookalike` | POST | Предсказание аудитории |
| `/lookalike/batch` | POST | Пакетное предсказание |
| `/model/info` | GET | Информация о модели |
| `/monitoring/drift` | GET | Статус дрейфа |
| `/monitoring/data-quality` | GET | Качество данных |
| `/experiments` | GET | История экспериментов |

## DVC Пайплайн

```bash
# Инициализация DVC
dvc init

# Запуск пайплайна
dvc repro

# Просмотр статуса
dvc status
```

## Тестирование

### Локальное тестирование на данных v1/v2

```bash
python tests/test_local.py --action all --version v1
```

### API тесты

```bash
pytest tests/test_api.py -v
```

## Технологии

- **FastAPI** - REST API
- **PyTorch + Implicit** - ALS модель для collaborative filtering
- **Great Expectations** - Валидация данных
- **Evidently** - Детекция дрейфа
- **MLflow** - Трекинг экспериментов
- **DVC** - Версионирование данных и моделей
- **MinIO** - S3-совместимое хранилище

## Архитектура

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Client    │────▶│  FastAPI     │────▶│   Model     │
│             │◀────│  (port 80)   │◀────│   (ALS)     │
└─────────────┘     └──────────────┘     └─────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   Pipeline   │
                    │   (async)    │
                    └──────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│    MinIO     │  │    MLflow    │  │     DVC      │
│   (S3 data)  │  │  (tracking)  │  │  (version)   │
└──────────────┘  └──────────────┘  └──────────────┘
```

## Лицензия

CC BY-NC-SA 4.0
