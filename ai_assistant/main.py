# main.py
import asyncio
import sys
import os

from ai_assistant.central_ai.init import CentralAISystem
from ai_assistant.config import CONFIG
from ai_assistant.utils.file_handler import FileHandler
from ai_assistant.utils.data_preprocessor import DataPreprocessor

CONFIG = {
    'central_ai': {
        'input_size': 512,
        'hidden_size': 768,
        'output_size': 256,
        'num_heads': 8,
        'num_layers': 6
    },
    'task_router': {
        'num_intents': 10
    },
    'learning_engine': {
        'learning_rate': 1e-4,
        'batch_size': 32
    }
}

async def main():
    # Initialize components
    file_handler = FileHandler(base_dir='./data')
    data_preprocessor = DataPreprocessor()

    # Initialize the Central AI System
    central_ai_system = CentralAISystem(CONFIG)
    await central_ai_system.initialize()

    # Load and preprocess some sample data
    sample_data = file_handler.read_file('./data/sample_data.csv')
    preprocessed_data = data_preprocessor.process_data(sample_data, {
        'numeric_columns': ['feature1', 'feature2'],
        'categorical_columns': ['category'],
        'text_columns': ['text_input'],
        'handle_missing': True,
        'imputation_strategy': 'mean'
    })

    # Example usage of the system
    user_input = "Generate an image of a sunset"
    result = await central_ai_system.process_input(user_input)
    print("AI Response:", result)

    # Get system stats
    stats = await central_ai_system.get_system_stats()
    print("System Stats:", stats)

    # Fine-tune a model
    await central_ai_system.fine_tune_model("conversation_ai", preprocessed_data)

    # Save system state
    await central_ai_system.save_system_state('system_state.pth')

if __name__ == "__main__":
    asyncio.run(main())