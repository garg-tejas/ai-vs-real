from torchvision.models import efficientnet_b4
import torch.nn as nn


def build_efficientnet(num_classes):
    model = efficientnet_b4(weights="DEFAULT")
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model