# base_ai.py
from abc import ABC, abstractmethod
import torch

class BaseAI(ABC):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def process(self, input_data):
        pass

    @abstractmethod
    def get_capabilities(self):
        pass

    def fine_tune(self, dataset):
        raise NotImplementedError("Fine-tuning not implemented for this model")

    def save_model(self, path):
        if self.model:
            torch.save(self.model.state_dict(), path)

    def load_saved_model(self, path):
        if self.model:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.model.to(self.device)