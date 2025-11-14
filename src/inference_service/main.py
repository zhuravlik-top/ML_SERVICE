from fastapi import FastAPI, HTTPException
from src.inference_service.model_loader import ModelRegistry
from src.inference_service.api_schemas import PredictRequest, PredictResponse, HealthResponse
from src.common.preprocessing import preprocess
from src.common.logger import get_logger
from dotenv import load_dotenv
import os

# Загружаем переменные окружения из .env
load_dotenv()

# Получаем путь к папке с моделями и к файлу registry
MODEL_DIR = os.getenv("MODEL_DIR", "./models")
REGISTRY_PATH = os.getenv("MODEL_REGISTRY_PATH", "./models/registry/metadata.json")

# Инициализируем FastAPI-приложение
app = FastAPI(title="ML Inference Service")

# Создаём логгер для этого модуля
logger = get_logger("InferenceService")

# Загружаем модель через ModelRegistry
registry = ModelRegistry(REGISTRY_PATH, MODEL_DIR)

# ----------------- Эндпоинт Health -----------------
@app.get("/health", response_model=HealthResponse)
def health():
    """
    Проверяет состояние сервиса.
    Возвращает:
        - статус работы сервиса
        - загружена ли модель
        - текущая модель
    """
    logger.info("Health check requested")
    return HealthResponse(
        status="ok",
        model_loaded=registry.model is not None,
        current_model=registry.current_model or "None"
    )

# ----------------- Эндпоинт Predict -----------------
@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Принимает текст, преобразует в эмбеддинг и возвращает предсказанный класс.
    """
    if registry.model is None:
        logger.warning("Prediction requested but model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        # Преобразуем входные данные в DataFrame
        embedding = preprocess(request.text)
        # Делаем предсказание
        prediction = registry.model.predict(embedding)[0]
        logger.info(f"Prediction made: {prediction}")
        return PredictResponse(topic=str(prediction))
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ----------------- Эндпоинт Reload Model -----------------
@app.post("/reload_model")
def reload_model():
    """
    Перезагружает модель из registry.
    Используется при обновлении модели.
    """
    registry.reload_model()
    logger.info(f"Model reloaded: {registry.current_model}")
    return {"status": "model reloaded", "current_model": registry.current_model}
