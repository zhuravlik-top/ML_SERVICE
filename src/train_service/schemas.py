from pydantic import BaseModel
from typing import Optional

class TrainResponse(BaseModel):
    status: str

    model_name: Optional[str] = None

    f1_macro: Optional[float] = None

    detail: Optional[str] = None

class TrainStatus(BaseModel):
    status : str

    last_model: Optional[str] = None

    best_f1: Optional[float] = None