# ai_registry.py
import importlib
import os

class AIRegistry:
    def __init__(self):
        self.ai_models = {}

    def register(self, name, ai_class):
        if not issubclass(ai_class, BaseAI):
            raise ValueError(f"{ai_class.__name__} must inherit from BaseAI")
        self.ai_models[name] = ai_class()
        self.ai_models[name].load_model()

    def get_ai(self, name):
        return self.ai_models.get(name)

    def list_ais(self):
        return list(self.ai_models.keys())

    def load_from_file(self, file_path):
        module_name = os.path.splitext(os.path.basename(file_path))[0]
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        for name, obj in module.__dict__.items():
            if isinstance(obj, type) and issubclass(obj, BaseAI) and obj != BaseAI:
                self.register(name, obj)

    def load_from_folder(self, folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.py') and not file_name.startswith('__'):
                file_path = os.path.join(folder_path, file_name)
                self.load_from_file(file_path)

    def fine_tune_model(self, name, dataset):
        if name in self.ai_models:
            self.ai_models[name].fine_tune(dataset)
        else:
            raise ValueError(f"No AI model named {name} found")