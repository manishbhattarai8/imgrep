import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class ResNetImageEncoder(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        resnet = models.resnet50(weights=None)  # No pretraining
        modules = list(resnet.children())[:-1]  # remove final fc layer
        self.resnet = nn.Sequential(*modules)
        
        # Fix: Use the actual output dimension, not the ResNet's fc input dimension
        self.output_dim = output_dim  # This should be 512, not 2048
        self.fc = nn.Linear(resnet.fc.in_features, output_dim)  # 2048 -> 512

    def forward(self, x):
        x = self.resnet(x).squeeze()
        x = self.fc(x)
        return x