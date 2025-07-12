import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageTextRetrievalModel(nn.Module):
    def __init__(self, embed_dim, image_encoder, text_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.image_proj = nn.Linear(image_encoder.output_dim, embed_dim)
        self.text_proj = nn.Linear(text_encoder.output_dim, embed_dim)

    def forward(self, images, captions):
        image_features = self.image_proj(self.image_encoder(images))
        text_features = self.text_proj(self.text_encoder(captions))
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        return image_features, text_features