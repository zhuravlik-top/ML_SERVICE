
---

## Описание ключевых файлов

### Train Service

- `main_train.py` — основной скрипт запуска обучения. Загружает данные, предобрабатывает их , запускает обучение модели и обновляет реестр моделей.  
- `trainer.py` — функция `train_model(df, model_dir, registry_path)` для обучения модели и деплоя.  
- `utils.py` — вспомогательные функции:
  - `save_model()` — сохраняет модель на диск;  
  - `load_preprocess_data()` / `save_preprocess_data()` — работа с эмбеддингами и метками;   
- `schemas.py` — Pydantic-схемы для API: `TrainResponse`, `TrainStatus`.
- `deploy.py` —  - `update_registry()` — обновляет `metadata.json` с информацией о новой модели. .

### Inference Service

- `main.py` — точка входа FastAPI для инференса. Сервис загружает модель из реестра и предоставляет API.  
- `model_loader.py` — класс `ModelRegistry`, который управляет загрузкой и хранением текущей модели.  
- `api_schemas.py` — Pydantic-схемы для API инференса: `PredictRequest`, `PredictResponse`, `HealthResponse`.

### Common

- `preprocessing.py` — функции для преобразования текста в эмбеддинги .
-  `logger.py` — файл отвечающий за создание логов.


### Docker & Configuration

- `docker-compose.yml` — описание сервисов, сетей и томов для Docker.  
- `Dockerfile.train` — сборка контейнера Train Service.  
- `Dockerfile.inference` — сборка контейнера Inference Service.  
- `.env` — переменные окружения (пути к папкам с данными и моделями).  
- `.gitignore` и `.dockerignore` — игнорирование больших файлов: `models/`, `data/`, `.env`.

---

