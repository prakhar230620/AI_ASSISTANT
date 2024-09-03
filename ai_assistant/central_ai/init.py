# __init__.py
import asyncio
from ai_assistant.central_ai.core import CentralAISystem

async def initialize_central_ai(config):
    return await CentralAISystem.create(config)

def run_central_ai(config):
    return asyncio.run(initialize_central_ai(config))
