import os
import json
import joblib  # для сохранения и загрузки моделей sklearn
from ..common.logger import get_logger

logger = get_logger("ModelRegistry")

class ModelRegistry:
    """
    Класс для работы с ML-моделью и registry.
    Хранит текущую модель в памяти, позволяет её перезагружать.
    """

    def __init__(self, registry_path: str, model_dir: str):
        self.registry_path = registry_path  # путь к JSON-файлу registry
        self.model_dir = model_dir          # папка, где лежат модели
        self.model = None                   # сама модель (объект sklearn, joblib и т.д.)
        self.current_model = None           # имя текущей модели (файл)
        self.load_model()                   # загружаем модель при старте

    def load_model(self):
        """
        Загружает модель из registry и помещает её в память.
        """
        logger.info("Loading model from registry")
        if not os.path.exists(self.registry_path):
            logger.warning(f"Registry file not found: {self.registry_path}")
            return

        # Открываем JSON registry
        with open(self.registry_path, "r") as f:
            metadata = json.load(f)

        # Берём имя текущей модели
        model_file = metadata.get("current_model")
        if model_file:
            model_path = os.path.join(self.model_dir, model_file)
            if os.path.exists(model_path):
                try:
                    # Загружаем модель в память
                    self.model = joblib.load(model_path)
                    self.current_model = model_file
                    logger.info(f"Loaded model: {model_file}")
                except Exception as e:
                    logger.error(f"Failed to load model {model_file}: {e}")
                    self.model = None
            else:
                logger.warning(f"Model file not found: {model_path}")
        else:
            logger.warning("No current_model specified in registry")

    def reload_model(self):
        """
        Перезагружает модель из registry.
        Используется при обновлении модели.
        """
        logger.info("Reloading model")
        self.load_model()
