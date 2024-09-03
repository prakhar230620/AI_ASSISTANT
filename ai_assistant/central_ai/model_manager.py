# model_manager.py
from typing import Dict, List, Any
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelVersion:
    def __init__(self, model: Any, version: int, timestamp: datetime):
        self.model = model
        self.version = version
        self.timestamp = timestamp
        self.performance: Dict[str, float] = {}

class ModelManager:
    def __init__(self):
        self.models: Dict[str, List[ModelVersion]] = {}
        self.model_usage: Dict[str, int] = {}
        self.fallback_model: Any = None
        self.learning_engine = None

    async def register_model(self, model_name: str, model: Any):
        if model_name not in self.models:
            self.models[model_name] = []

        new_version = len(self.models[model_name]) + 1
        model_version = ModelVersion(model, new_version, datetime.now())
        self.models[model_name].append(model_version)
        self.model_usage[model_name] = 0

        logger.info(f"Model {model_name} (v{new_version}) registered successfully.")

    async def get_model(self, model_name: str) -> Any:
        if model_name not in self.models or not self.models[model_name]:
            raise ValueError(f"Model {model_name} does not exist.")

        self.model_usage[model_name] += 1
        return self.models[model_name][-1].model

    async def update_model_performance(self, model_name: str, y_true: List[Any], y_pred: List[Any]):
        if model_name not in self.models or not self.models[model_name]:
            raise ValueError(f"Model {model_name} does not exist.")

        latest_version = self.models[model_name][-1]
        latest_version.performance = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average='weighted'),
            "recall": recall_score(y_true, y_pred, average='weighted'),
            "f1": f1_score(y_true, y_pred, average='weighted')
        }

        logger.info(f"Performance metrics updated for model {model_name} (v{latest_version.version}).")

    async def get_all_models(self) -> Dict[str, Any]:
        return {name: versions[-1].model for name, versions in self.models.items()}

    def set_learning_engine(self, learning_engine):
        self.learning_engine = learning_engine