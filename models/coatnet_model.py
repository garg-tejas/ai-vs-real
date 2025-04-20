from timm import create_model

def build_coatnet(num_classes):
    model = create_model("coatnet_0_rw_224", pretrained=True, num_classes=num_classes)
    return model