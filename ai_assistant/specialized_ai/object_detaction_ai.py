# object_detection_ai.py
from .base_ai import BaseAI
import torch
import torchvision


class ObjectDetectionAI(BaseAI):
    def load_model(self):
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.to(self.device)
        self.model.eval()

    def process(self, input_data):
        # Assuming input_data is a PIL Image
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        img = transform(input_data).unsqueeze(0).to(self.device)

        with torch.no_grad():
            prediction = self.model(img)

        return prediction[0]

    def get_capabilities(self):
        return ["object_detection", "image_analysis"]

    def fine_tune(self, dataset):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

        for epoch in range(10):  # You can adjust the number of epochs
            for images, targets in dataset:
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

            print(f"Epoch {epoch + 1} completed")


# Register the AI
from . import registry

registry.register("ObjectDetectionAI", ObjectDetectionAI)