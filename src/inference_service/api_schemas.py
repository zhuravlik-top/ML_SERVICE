from pydantic import BaseModel
from typing import List

class PredictRequest(BaseModel):
    text: str  # вход — один текст

class PredictResponse(BaseModel):
    topic: str  # выход — предсказанная тема


class HealthResponse(BaseModel):
    """
    Модель ответа эндпоинта /health.
    Показывает состояние сервиса и текущую загруженную модель.
    """
    status: str          # например "ok"
    model_loaded: bool   # True, если модель загружена
    current_model: str   # имя файла модели
