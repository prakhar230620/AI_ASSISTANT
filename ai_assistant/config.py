# config.py

CONFIG = {
    'central_ai': {
        'input_size': 512,
        'hidden_size': 768,
        'output_size': 256,
        'num_heads': 8,
        'num_layers': 6
    },
    'specialized_models': [
        {'name': 'conversation_ai', 'type': 'gpt2', 'path': 'models/conversation_ai.pth'},
        {'name': 'image_generation_ai', 'type': 'gan', 'path': 'models/image_generation_ai.pth'},
        {'name': 'object_detection_ai', 'type': 'fasterrcnn', 'path': 'models/object_detection_ai.pth'}
    ],
    'task_router': {
        'num_intents': 10
    },
    'learning_engine': {
        'learning_rate': 1e-4,
        'batch_size': 32
    }
}