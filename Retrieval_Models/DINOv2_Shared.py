import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

class get_MINIMA_Roma_DINOv2_model(nn.Module):
    def __init__(self, dinov2_backbone, p=4.0):
        """"
            p is the exponenet in GeM pooling
        """
        super().__init__()
        self.model = dinov2_backbone 
        self.p = nn.Parameter(torch.ones(1) * p)

    def forward(self, x, return_dense=False):
        with torch.inference_mode():
            weight_dtype = next(self.model.parameters()).dtype 
            
            x = x.to(weight_dtype)

            features_dict = self.model.forward_features(x)
            
            patch_features = features_dict["x_norm_patchtokens"]

            # GeM Pooling 
            x_clamped = patch_features.clamp(min=1e-6)
            x_pow = x_clamped.pow(self.p)
            x_mean = x_pow.mean(dim=1)
            global_feat = x_mean.pow(1.0 / self.p)
            global_feat = F.normalize(global_feat, p=2, dim=-1)
            if return_dense:
                cls_token = features_dict["x_norm_clstoken"] # [B, D]
                cls_token = F.normalize(cls_token, p=2, dim=-1)
                return global_feat, patch_features, cls_token
            return global_feat

def get_MINIMA_Roma_DINOv2_transform(resize_size = 518):
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])