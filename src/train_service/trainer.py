import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from src.train_service.utils import make_model_name, save_model, read_json, load_preprocess_data
from src.train_service.deploy import update_registry
from src.common.preprocessing import preprocess
import os

def train_model(df:pd.DataFrame, model_dir:str, registry_peth:str, model_prefix: str = 'model', preprocess_dir="/app/data/processed"):
    """
    Oбучение и деплой модели
    Возвращает словарь с результатом: {"status": "trained", "model_name": ..., "f1_macro": ...}
    """
    # Проверка входных данных
    #if "text" not in df.columns or "label" not in df.columns:
        #raise ValueError("DataFrame must contain 'text' and 'label' columns")
    
    #texts = df['text'].astype(str).tolist()
    
    #X=preprocess(texts)

    #y = df['label'].tolist()
    X, y = load_preprocess_data(preprocess_dir)


    X_train, X_val, y_train, y_val = train_test_split(X ,y, test_size=0.2, random_state=42, stratify=y)
    base_model = LogisticRegression(max_iter=3000, random_state=42)

    param_grid = {
    'C': [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
    'solver': ['lbfgs', 'newton-cg', 'sag'],
    'penalty': ['l2', None]
    }
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring='f1_macro',
        cv = 3,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    y_pred = best_model.predict(X_val)
    f1_val = float(f1_score(y_val, y_pred, average="macro"))

    model_name = make_model_name(model_prefix)
    print(f"[TRAIN] Generated model name: {model_name}")

    model_path = save_model(best_model, model_dir, model_name)
    print(f"[TRAIN] Model saved at: {model_path}")


    # Читаем текущий registry
    registry = read_json(registry_peth) or {"current_model": None, "models": []}
    print(f"[TRAIN] Loaded registry: {registry}")
    current_model_name = registry.get('current_model')
    current_f1 = 0.0

    if current_model_name:
        for m in registry.get('models', []):
            if m['name'] == current_model_name:
                current_f1 = m.get("metrics", {}).get("f1_macro", 0.0)
                break

    # Новая модель станет current только если f1_val >= текущей
    update_current = f1_val >= current_f1
    print(f"[TRAIN] Should update current?: {update_current} (new f1={f1_val}, old f1={current_f1})")

    update_registry(model_dir, registry_peth, model_name, 
        metrics={
        "f1_macro": f1_val,
        'best_params': best_params},
        update_current=update_current
    )
    print(f"[TRAIN] Registry updated.")
    return {
        "status": "trained",
        "model_name": model_name,
        "f1_macro": f1_val,
        "best_params": best_params
    }