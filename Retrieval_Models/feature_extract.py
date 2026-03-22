import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.cuda.amp import autocast
import torch.nn.functional as F

import torch
import torch.nn.functional as F
import numpy as np

def compute_shift_from_heatmap(heatmap, grid_size=None, base_gain=1.0, max_shift_limit=0.6):
    """
      SGVA Module: Compute uncertainty-aware spatial shift and elastic scale from a similarity heatmap.
    """
    B, N = heatmap.shape
    device = heatmap.device
    
    if grid_size is None:
        grid_size = int(np.sqrt(N))
    
    heatmap_2d = heatmap.reshape(B, grid_size, grid_size)
    
    # ReLu and Normalization
    heatmap_2d = F.relu(heatmap_2d)
    heatmap_sum = heatmap_2d.sum(dim=(1, 2), keepdim=True) + 1e-6
    heatmap_2d = heatmap_2d / heatmap_sum
    
    # get the norm grid
    y_coords, x_coords = torch.meshgrid(
        torch.arange(grid_size, device=device, dtype=torch.float32), 
        torch.arange(grid_size, device=device, dtype=torch.float32),
        indexing='ij'
    )
    x_norm = x_coords / (grid_size - 1)
    y_norm = y_coords / (grid_size - 1)
    
    x_norm = x_norm.expand(B, -1, -1)
    y_norm = y_norm.expand(B, -1, -1)
    
    results = []
    for i in range(B):
        h_map = heatmap_2d[i]
        total_mass = h_map.sum() + 1e-9
        
        # Center of Gravity
        cog_x = (h_map * x_norm[i]).sum() / total_mass
        cog_y = (h_map * y_norm[i]).sum() / total_mass
        
        # Spatial Variance / Uncertainty
        dist_sq = (x_norm[i] - cog_x)**2 + (y_norm[i] - cog_y)**2
        variance = (h_map * dist_sq).sum() / total_mass
        std_dev = torch.sqrt(variance)
        uncertainty_score = torch.clamp(std_dev * 5.0, 0.0, 1.0) 
        adaptive_gain = base_gain * (1.0 + 0.5 * (1.0 - uncertainty_score))
        
        scale_factor = 1.0 + 0.2 * uncertainty_score
        
        # Shift calculation. The normalize center is 0.5
        shift_x = (cog_x - 0.5) * adaptive_gain
        shift_y = (cog_y - 0.5) * adaptive_gain
        
        shift_x = torch.clamp(shift_x, -max_shift_limit, max_shift_limit)
        shift_y = torch.clamp(shift_y, -max_shift_limit, max_shift_limit)
        
        results.append((shift_x.item(), shift_y.item(), scale_factor.item()))
        
    return results

def extract_features(retrieval_size, config, method_dict, MID, block_size, SATELLITE, UAV, val_transforms, model):
    """
        Feature Extraction
    """
    
    model.eval()
    # UAV global feature extraction
    uav_global_feat = None
    with torch.no_grad():
        with autocast():
            UAV1 = UAV.to(config['DEVICE']).unsqueeze(0)
            
            if 'DINOv2' in method_dict['retrieval_method']:
                uav_global_feat, _, uav_cls_token = model(UAV1, return_dense=True)
                if config['RETRIEVAL_FEATURE_NORM']:
                    uav_global_feat = F.normalize(uav_global_feat, dim=-1)
                    uav_cls_token = F.normalize(uav_cls_token, dim=-1)
            else:
                uav_global_feat = model(UAV1)
                uav_global_feat = feature_fusion_all(method_dict, uav_global_feat)
                if config['RETRIEVAL_FEATURE_NORM']:
                    uav_global_feat = F.normalize(uav_global_feat, dim=-1)

    row, column = MID.shape
    process = list(range(row))
    img_features_list = []
    all_shifts = []
    
    mids = MID.reshape(-1,2)
    print(mids.shape) # satellite block number
    # satellite global feature extraction
    for batch_start in tqdm(range(0, len(process), config['BATCH_SIZE']), desc='Image Retrieval'):

        # Get the end index of the current batch
        batch_end = batch_start + config['BATCH_SIZE']
        # If the batch end index exceeds the total number, use the actual number
        if batch_end > len(process):
            batch_end = len(process)

        batch_indices = process[batch_start:batch_end]
        batch_images = []

        for item in batch_indices:
            mid_x, mid_y = mids[item, :]
            mid_x, mid_y = mids[item, :]
            img = cv2.resize(SATELLITE[int(mid_x - block_size[0] / 2):int(mid_x + block_size[0] / 2),
                  int(mid_y - block_size[1] / 2):int(mid_y + block_size[1] / 2)], (retrieval_size,retrieval_size))
            img = Image.fromarray(img)
            img = val_transforms(img)
            batch_images.append(img)

        if len(batch_images) > 0:
            batch_images = torch.stack(batch_images, dim=0).to(config['DEVICE'])
            
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    
                    if 'DINOv2' in method_dict['retrieval_method']:
                        sat_global, sat_dense, _ = model(batch_images, return_dense=True)
                        sat_dense_norm = F.normalize(sat_dense, p=2, dim=-1)
                        query = uav_cls_token.unsqueeze(-1)
                        
                        # compute similarity map
                        similarity_map = torch.matmul(sat_dense_norm, query).squeeze(-1) # [B, N]
                        
                        # compute shifts
                        batch_shifts = compute_shift_from_heatmap(similarity_map)
                        
                        all_shifts.extend(batch_shifts)
                        
                        batch_features = sat_global
                        
                    else:
                        batch_features = model(batch_images)
                        batch_features = feature_fusion_all(method_dict, batch_features)
                        all_shifts.extend([(0.0, 0.0)] * len(batch_images))
                    if config['RETRIEVAL_FEATURE_NORM']:
                        batch_features = F.normalize(batch_features, dim=-1)
                        
            img_features_list.append(batch_features.to(torch.float32))
        else:
             pass

    if len(img_features_list) > 0:
        img_features = torch.cat(img_features_list, dim=0)
    else:
        img_features = torch.tensor([], dtype=torch.float32, device=config['DEVICE'])

    return img_features, uav_global_feat, all_shifts

def feature_fusion_all(method_dict, img_feature):
    if method_dict['retrieval_method'] == 'MCCG':
        if len(img_feature.shape) == 3:
            fnorm = torch.norm(img_feature, p=2, dim=1, keepdim=True) * np.sqrt(img_feature.size(-1))
            img_feature = img_feature.div(fnorm.expand_as(img_feature))
            img_feature = img_feature.view(img_feature.size(0), -1)
        else:
            fnorm = torch.norm(img_feature, p=2, dim=1, keepdim=True)
            img_feature = img_feature.div(fnorm.expand_as(img_feature))
    elif method_dict['retrieval_method'] == 'LPN':
        block = 6
        fnorm = torch.norm(img_feature, p=2, dim=1, keepdim=True) * np.sqrt(block)
        img_feature = img_feature.div(fnorm.expand_as(img_feature))
        img_feature = img_feature.view(img_feature.size(0), -1)
    elif method_dict['retrieval_method'] == 'MIFT':
        if len(img_feature.shape)==3:
            fnorm = torch.norm(img_feature, p=2, dim=1, keepdim=True) * np.sqrt(img_feature.size(-1))
            img_feature = img_feature.div(fnorm.expand_as(img_feature))
            img_feature = img_feature.view(img_feature.size(0), -1)
        else:
            fnorm = torch.norm(img_feature, p=2, dim=1, keepdim=True)
            img_feature = img_feature.div(fnorm.expand_as(img_feature))
    elif method_dict['retrieval_method'] == 'FSRA':
        if len(img_feature.shape)==3:
            fnorm = torch.norm(img_feature, p=2, dim=1, keepdim=True) * np.sqrt(img_feature.size(-1))
            img_feature = img_feature.div(fnorm.expand_as(img_feature))
            img_feature = img_feature.view(img_feature.size(0), -1)
        else:
            fnorm = torch.norm(img_feature, p=2, dim=1, keepdim=True)
            img_feature = img_feature.div(fnorm.expand_as(img_feature))

    elif method_dict['retrieval_method'] == 'CAMP':
            img_feature = img_feature[-2]
    elif method_dict['retrieval_method'] == 'DAC':
            img_feature = img_feature[-2]
    return img_feature
