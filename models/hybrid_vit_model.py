from timm import create_model

def build_hybrid_vit(num_classes):
    model = create_model("vit_base_r50_s16_224.orig_in21k", pretrained=True, num_classes=num_classes)
    return model