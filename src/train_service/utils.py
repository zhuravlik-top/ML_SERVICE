import os
import json
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
from src.common.preprocessing import preprocess

def make_model_name(prefix: str = 'model'):
    """
    Генерирует уникальное имя модели по текущему времени.
    Пример результата: "model_20251111_203045.pkl"
    """
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{prefix}_{ts}.pkl"

def save_model(model, model_dir:str, model_name:str):
    """
    Сохраняет модель на диск с помощью joblib.
    Возвращает абсолютный путь к сохранённому файлу.
    """
    print(f"[SAVE_MODEL] Saving model to {model_dir}/{model_name}")
    path = os.path.join(model_dir, model_name)

    joblib.dump(model, path)
    print(f"[SAVE_MODEL] Saved OK.")

    return path

def read_json(path:str):
    """
    Безопасно читает JSON-файл и возвращает распарсенные данные (dict/list).
    Если файл не существует — возвращает None.
    """
    if not os.path.exists(path):
        return None
    
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def write_json(path:str, data:dict):
    """
    Атомарно записывает словарь data в JSON-файл по пути path.
    Схема:
      1. Пишем содержимое в временный файл path + ".tmp"
      2. Перемещаем (os.replace) временный файл на место path — это атомарная операция на большинстве ОС
    Это уменьшает шанс того, что registry останется в частично записанном состоянии.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    tmp = path + '.tmp'

    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    os.replace(tmp, path)

def load_raw_data(raw_dir:str)->pd.DataFrame:
    """
    Читает все JSON файлы из raw/ и возвращает DataFrame с колонками 'text' и 'label'.
    """
    all_data=[]
    for fname in os.listdir(raw_dir):
        if fname.endswith('.json'):
            with open(os.path.join(raw_dir, fname), 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_data.extend(data)
    return pd.DataFrame(all_data)

def save_preprocess_data(df:pd.DataFrame, preprocess_dir:str):
    """
    Применяет preprocess к колонке text и сохраняет X и y в npy.
    """
    os.makedirs(preprocess_dir, exist_ok=True)
    texts = df['text'].tolist()
    X = preprocess(texts)
    y = np.array(df["label"].tolist())

    np.save(os.path.join(preprocess_dir, "X.npy"), X)
    np.save(os.path.join(preprocess_dir, "y.npy"), y)

def load_preprocess_data(preprocess_dir:str):
    """
    Загружает эмбеддинги и метки из preprocess_dir.
    """
    X = np.load(os.path.join(preprocess_dir, "X.npy"))
    y = np.load(os.path.join(preprocess_dir, "y.npy"))
    return X, y