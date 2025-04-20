from torchvision.models import convnext_base
import torch.nn as nn

def build_convnext(num_classes):
    model = convnext_base(weights="DEFAULT")
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    return model