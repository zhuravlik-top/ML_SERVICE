import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

# Загружаем SentenceTransformer при старте
EMB_MODEL_NAME = "all-MiniLM-L6-v2"
st_model = SentenceTransformer(EMB_MODEL_NAME)

def preprocess(texts):
    """
    Преобразует список текстов в эмбеддинги с помощью SentenceTransformer.
    """
    if isinstance(texts, str):
        texts = [texts]

    embeddings = st_model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Если формат неизвестен — выбрасываем ошибку
    if not texts: 
        raise ValueError("Unsupported input format for preprocessing")

    return embeddings
