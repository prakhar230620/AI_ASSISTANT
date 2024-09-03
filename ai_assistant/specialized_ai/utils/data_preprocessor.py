# utils/data_preprocessor.py
import torch
from torchvision import transforms
from PIL import Image


def preprocess_image(image_path, target_size=(224, 224)):
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)


def preprocess_text(text, tokenizer, max_length=512):
    return tokenizer(text, padding='max_length', max_length=max_length, truncation=True, return_tensors='pt')