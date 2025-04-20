from torchvision.models import resnet50
import torch.nn as nn

def build_resnet(num_classes):
    model = resnet50(weights="DEFAULT")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model