# 文件路径: Retrieval_Models/DINOv3.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import sys
import os
DINOv3_ROOT = os.path.dirname(os.path.abspath(__file__))
if DINOv3_ROOT not in sys.path:
    sys.path.append(DINOv3_ROOT)
class get_DINOv3_model(nn.Module):
    def __init__(self, p=4.0):
        super().__init__()
        
        self.repo_dir = DINOv3_ROOT
        # self.weights_path = os.path.join(self.repo_dir, 'weights/backbone/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth')
                    
        print(f"Loading DINOv3 model from: {self.repo_dir}")
        # print(f"Loading weights from: {self.weights_path}")
    
        self.model = torch.hub.load(
            self.repo_dir,
            'dinov3_vitl16',       
            source='local',            
            # weights=self.weights_path, 
        )

        self.p = nn.Parameter(torch.ones(1) * p)

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x, return_dense=False): 
        with torch.inference_mode():
            weight_dtype = next(self.model.parameters()).dtype 
            x = x.to(weight_dtype)
            
            features_dict = self.model.forward_features(x)
            
            patch_features = features_dict["x_norm_patchtokens"]
            
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

def get_DINOv3_transform(resize_size=384):
    return T.Compose([
        T.Resize((resize_size, resize_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

