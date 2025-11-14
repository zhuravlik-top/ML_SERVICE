import os
from datetime import datetime
from src.train_service.utils import read_json, write_json

def update_registry(model_dir:str, registry_path:str, model_name:str, metrics:dict, update_current:bool = True):
    """
    Добавляет запись о новой модели в registry (metadata.json) и делает её current_model.
    Возвращает:
      Обновлённый registry (словарь), как было записано в metadata.json
    """
    model_path = os.path.join(model_dir, model_name)

    if not os.path.exists(model_path):
        #бросаем ошибку, чтобы вызывающий код понял, что деплой не удался.
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    registry = read_json(registry_path) or {"current_model": None, "models": []}

    # Добавляем новую запись о модели в список "models".
    registry['models'].append({
        'name':model_name, 
        'metrics':metrics or {}, 
        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

    # Обновляем поле current_model — теперь сервисы должны использовать эту модель при старте/перезагрузке.
    if update_current:
        registry['current_model'] = model_name

    # Записываем обновлённый registry обратно на диск
    write_json(registry_path, registry)

    return registry