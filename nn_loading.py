from torchvision import models
import torch.nn as nn

class MultiLabelResnet18(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, n_classes),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        out = self.resnet(x)
        return out
