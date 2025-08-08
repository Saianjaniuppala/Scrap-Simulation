# src/model.py
import torch
import torch.nn as nn
import torchvision.models as models

def get_resnet18(num_classes, pretrained=True):
    model = models.resnet18(pretrained=pretrained)
    # Replace final fc
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
