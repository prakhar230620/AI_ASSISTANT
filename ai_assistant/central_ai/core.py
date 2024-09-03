# core.py
import logging
from typing import Dict, Any
import torch
import asyncio

from .task_router import TaskRouter
from .model_manager import ModelManager
from .learning_engine import LearningEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CentralAISystem:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self.model_manager = None
        self.task_router = None
        self.learning_engine = None

    @classmethod
    async def create(cls, config: Dict[str, Any]):
        instance = cls(config)
        await instance.initialize()
        return instance

    async def initialize(self):
        self.model_manager = await self._initialize_model_manager()
        self.task_router = await self._initialize_task_router()
        self.learning_engine = self._initialize_learning_engine()

        await self._integrate_components()
        logger.info("Central AI System initialized successfully")

    async def _initialize_model_manager(self) -> ModelManager:
        logger.info("Initializing Model Manager")
        return ModelManager()

    async def _initialize_task_router(self) -> TaskRouter:
        logger.info("Initializing Task Router")
        specialized_ais = await self.model_manager.get_all_models()
        return TaskRouter(self, specialized_ais, num_intents=self.config['task_router']['num_intents'])

    def _initialize_learning_engine(self) -> LearningEngine:
        logger.info("Initializing Learning Engine")
        return LearningEngine(self, learning_rate=self.config['learning_engine']['learning_rate'],
                              batch_size=self.config['learning_engine']['batch_size'])

    async def _integrate_components(self):
        logger.info("Integrating system components")
        self.task_router.set_model_manager(self.model_manager)
        self.model_manager.set_learning_engine(self.learning_engine)

    async def process_input(self, user_input: str) -> str:
        logger.info(f"Processing user input: {user_input}")
        try:
            result = await self.task_router.route_task(user_input)
            await self.learning_engine.update_from_interaction(user_input, result)
            return result
        except Exception as e:
            logger.error(f"Error processing input: {str(e)}")
            return "I apologize, but I encountered an error while processing your request. Please try again."

    async def analyze_and_modify(self, result, original_input):
        # This is a placeholder for more complex analysis and modification
        return f"Analyzed and modified: {result}"