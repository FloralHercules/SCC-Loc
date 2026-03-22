
from Retrieval_Models.CAMP.get_CAMP import get_CAMP_model
from torchvision import transforms
from Retrieval_Models.DINOv3.get_DINOv3 import get_DINOv3_model, get_DINOv3_transform
from Retrieval_Models.DINOv2_Shared import get_MINIMA_Roma_DINOv2_model, get_MINIMA_Roma_DINOv2_transform
def get_transforms_new():
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return data_transforms

def get_Model(model_name, DINOv2_shared = None):
    if model_name == 'CAMP':
        model = get_CAMP_model()
        val_transforms = get_transforms_new()
    elif model_name == 'DINOv3':
        model = get_DINOv3_model()
        val_transforms = get_DINOv3_transform(resize_size=384)
    elif model_name == 'MINIMA_Roma_DINOv2': # this is the shared model from RoMa or MINIMA_{RoMa}
        model = get_MINIMA_Roma_DINOv2_model(DINOv2_shared)
        val_transforms = get_MINIMA_Roma_DINOv2_transform(resize_size=518)
    elif model_name == 'DINOv2':
        import torch
        print("[INFO] Loading standalone DINOv2 (ViT-L/14) from PyTorch Hub...")
        standalone_backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        # we use the logistic of MINIMA_{RoMa} DINOv2 branch to init the no shared DINOv2 retrieval model
        model = get_MINIMA_Roma_DINOv2_model(standalone_backbone)
        val_transforms = get_MINIMA_Roma_DINOv2_transform(resize_size=518)
        
    return model, val_transforms








