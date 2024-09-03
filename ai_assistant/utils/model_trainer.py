# model_trainer.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, Any, Tuple
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()
        self.criterion = self._get_loss_function()

    def _get_optimizer(self):
        if self.config['optimizer'] == 'adam':
            return optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        elif self.config['optimizer'] == 'sgd':
            return optim.SGD(self.model.parameters(), lr=self.config['learning_rate'], momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {self.config['optimizer']}")

    def _get_scheduler(self):
        return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=10)

    def _get_loss_function(self):
        if self.config['loss'] == 'cross_entropy':
            return nn.CrossEntropyLoss()
        elif self.config['loss'] == 'mse':
            return nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss function: {self.config['loss']}")

    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        train_loader = self._create_data_loader(X_train, y_train)
        val_loader = self._create_data_loader(X_val, y_val)

        best_val_loss = float('inf')
        for epoch in range(self.config['epochs']):
            train_loss = self._train_epoch(train_loader)
            val_loss, val_metrics = self._validate(val_loader)

            logger.info(f"Epoch {epoch + 1}/{self.config['epochs']}")
            logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            logger.info(f"Val Metrics: {val_metrics}")

            self.scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pth')

        return val_metrics

    def _create_data_loader(self, X: np.ndarray, y: np.ndarray) -> torch.utils.data.DataLoader:
        dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
        return torch.utils.data.DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True)

    def _train_epoch(self, train_loader: torch.utils.data.DataLoader) -> float:
        self.model.train()
        total_loss = 0
        for batch_X, batch_y in tqdm(train_loader, desc="Training"):
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def _validate(self, val_loader: torch.utils.data.DataLoader) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch_X, batch_y in tqdm(val_loader, desc="Validating"):
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        val_loss = total_loss / len(val_loader)
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, average='weighted'),
            'recall': recall_score(all_labels, all_preds, average='weighted'),
            'f1': f1_score(all_labels, all_preds, average='weighted')
        }
        return val_loss, metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            _, preds = torch.max(outputs, 1)
        return preds.cpu().numpy()

    def fine_tune(self, X: np.ndarray, y: np.ndarray, epochs: int = 5) -> Dict[str, float]:
        self.config['epochs'] = epochs
        return self.train(X, y)