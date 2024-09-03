# image_generation_ai.py
from .base_ai import BaseAI
import torch
from torch import nn
from torchvision.utils import save_image
import numpy as np

class SimpleGAN(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(SimpleGAN, self).__init__()
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

class ImageGenerationAI(BaseAI):
    def load_model(self):
        latent_dim = 100
        img_shape = (3, 64, 64)
        self.model = SimpleGAN(latent_dim, img_shape).to(self.device)

    def process(self, input_data):
        z = torch.randn(1, 100).to(self.device)
        generated_img = self.model(z)
        save_image(generated_img.data, "generated_image.png", normalize=True)
        return "Image generated and saved as 'generated_image.png'"

    def get_capabilities(self):
        return ["image_generation"]

    def fine_tune(self, dataset):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0002, betas=(0.5, 0.999))
        criterion = nn.BCELoss()

        for epoch in range(200):
            for i, (imgs, _) in enumerate(dataset):
                valid = torch.ones(imgs.size(0), 1).to(self.device)
                fake = torch.zeros(imgs.size(0), 1).to(self.device)

                real_imgs = imgs.to(self.device)
                z = torch.randn(imgs.size(0), 100).to(self.device)
                gen_imgs = self.model(z)

                g_loss = criterion(self.discriminator(gen_imgs), valid)
                g_loss.backward()
                optimizer.step()

            if epoch % 50 == 0:
                print(f"Epoch {epoch}/{200}")

# Register the AI
from . import registry
registry.register("ImageGenerationAI", ImageGenerationAI)