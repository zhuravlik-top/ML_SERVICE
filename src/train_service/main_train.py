import os
import logging
from dotenv import load_dotenv
import pandas as pd
from src.train_service.trainer import train_model
from src.train_service.utils import load_preprocess_data, load_raw_data, save_preprocess_data

# ----------------- Настройка логирования -----------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("TrainService")

# ----------------- Загружаем переменные окружения -----------------
load_dotenv()
MODEL_DIR = os.getenv('MODEL_DIR', '/app/models')
REGISTRY_PATH = os.getenv('MODEL_REGISTRY_PATH', '/app/models/registry/metadata.json')
RAW_DIR = os.getenv("RAW_DATA_DIR", "/app/data/raw")
PREPROCESS_DIR = os.getenv('PROCESSED_DATA_DIR', '/app/data/processed')

# ----------------- Основная логика обучения -----------------
def main():
    logger.info("Train service started")

    # ----------------- Пересчёт preprocessed данных -----------------
    try:
        logger.info(f"Preprocessing all data from {RAW_DIR} into {PREPROCESS_DIR}")
        df_raw = load_raw_data(RAW_DIR)
        save_preprocess_data(df_raw, PREPROCESS_DIR)
        logger.info(f"Preprocessed data saved: X.npy and y.npy in {PREPROCESS_DIR}")
    except Exception as e:
        logger.exception(f"Failed to prepare preprocessed data: {e}")
        return

    # ----------------- Загружаем эмбеддинги и метки -----------------
    try:
        X, y = load_preprocess_data(PREPROCESS_DIR)
        logger.info(f"Loaded preprocessed data: X.shape={X.shape}, y.shape={y.shape}")
    except Exception as e:
        logger.exception(f"Failed to load preprocessed data: {e}")
        return

    # ----------------- Создаём DataFrame для train_model -----------------
    df_train = pd.DataFrame({"X": list(X), "y": y})

    # ----------------- Запуск обучения -----------------
    try:
        result = train_model(df_train, MODEL_DIR, REGISTRY_PATH)
        logger.info(f"Training completed. Model saved: {result['model_name']}, f1_macro={result['f1_macro']}")
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        return

if __name__ == "__main__":
    main()
