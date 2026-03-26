FROM python:3.11-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Обновление pip
RUN pip install --upgrade pip setuptools wheel

# Копирование requirements
COPY requirements.txt .

# Установка основных зависимостей (без dvc)
RUN pip install --no-cache-dir -r requirements.txt

# Установка DVC отдельно (может требовать другие версии)
RUN pip install --no-cache-dir "dvc[s3]>=3.30.0"

# Копирование кода приложения
COPY src/ ./src/
COPY params.yaml .

# Создание директорий для данных и моделей
RUN mkdir -p /app/data/raw /app/data/processed /app/models /app/reports

# Переменная окружения для MLflow
ENV MLFLOW_TRACKING_URI=http://mlflow:5000
ENV PYTHONPATH=/app

# Экспорт порта
EXPOSE 80

# Запуск приложения
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "80"]
