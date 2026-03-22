from PIL import Image
import cv2
import numpy as np
import os
import sys
import torch

# 添加RoMaV2模块到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from romav2 import RoMaV2


def RoMaV2_Init():

    print("Initializing RoMaV2 model...")
    
    cfg = RoMaV2.Cfg(compile=False)
    roma_v2_model = RoMaV2(cfg=cfg)
    roma_v2_model.apply_setting("precise")  # choose: "turbo", "fast", "base", "precise"
    print("RoMaV2 model initialized successfully.")
    return roma_v2_model


def RoMaV2_match(image0, image1, roma_v2_model, save_path=None, ransac_name="", need_ransac=False, show_matches=False):

    image0_origin = image0.copy()
    image1_origin = image1.copy()
    
    H_A, W_A = image0.shape[0], image0.shape[1]
    H_B, W_B = image1.shape[0], image1.shape[1]
    
    image0_rgb = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB)
    image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    
    image0_pil = Image.fromarray(image0_rgb)
    image1_pil = Image.fromarray(image1_rgb)
    
    with torch.inference_mode():
        preds = roma_v2_model.match(image0_pil, image1_pil)
    
    num_samples = 5000
    try:
        matches, conf, precision_A, precision_B = roma_v2_model.sample(preds, num_samples)
    except Exception as e:
        print(f"Warning: Failed to sample matches: {e}")
        matches = None
        conf = None
    
    if matches is not None:
        mkpts0_norm, mkpts1_norm = roma_v2_model.to_pixel_coordinates(
            matches.unsqueeze(0), H_A, W_A, H_B, W_B
        )
        mkpts0 = mkpts0_norm[0].cpu().numpy()
        mkpts1 = mkpts1_norm[0].cpu().numpy()
        mconf = conf.cpu().numpy()
    else:
        warp_AB = preds["warp_AB"][0]  
        overlap_AB = preds["overlap_AB"][0]
        
        H, W, _ = warp_AB.shape
        grid = torch.stack(
            torch.meshgrid(
                torch.linspace(-1, 1, H, device=warp_AB.device),
                torch.linspace(-1, 1, W, device=warp_AB.device),
                indexing='ij'
            ),
            dim=-1
        )
        
        conf_1d = overlap_AB.squeeze(-1).reshape(-1)
        
        threshold = 0
        valid_mask = conf_1d > threshold
        valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
        
        if len(valid_indices) > num_samples:
            sample_indices = torch.randperm(len(valid_indices))[:num_samples]
            valid_indices = valid_indices[sample_indices]
        
        valid_grid = grid.reshape(-1, 2)[valid_indices]
        valid_warp = warp_AB.reshape(-1, 2)[valid_indices]
        
        mkpts0 = ((valid_grid + 1) / 2) * torch.tensor([W - 1, H - 1], device=valid_grid.device)
        mkpts1 = ((valid_warp + 1) / 2) * torch.tensor([W - 1, H - 1], device=valid_warp.device)
        
        mkpts0 = mkpts0.cpu().numpy()
        mkpts1 = mkpts1.cpu().numpy()
        mconf = conf_1d[valid_indices].cpu().numpy()
    
    inliers_bool = np.ones(len(mkpts0), dtype=bool)
    
    if len(mkpts0) > 8 and need_ransac:
        src_pts = np.float32(mkpts0).reshape(-1, 1, 2)
        dst_pts = np.float32(mkpts1).reshape(-1, 1, 2)
        
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.USAC_MAGSAC, 3.0, maxIters=5000, confidence=0.9999)
        
        if mask is not None:
            inliers_bool = mask.ravel().astype(bool)
            mkpts0 = mkpts0[inliers_bool]
            mkpts1 = mkpts1[inliers_bool]
            mconf = mconf[inliers_bool]
    
    sort_indices = np.argsort(mconf)[::-1]
    mkpts0 = mkpts0[sort_indices]
    mkpts1 = mkpts1[sort_indices]
    mconf = mconf[sort_indices]
    
    Sen_pts = mkpts0.tolist()
    Ref_pts = mkpts1.tolist()
    mconf = mconf.tolist()
    
    if show_matches and len(Sen_pts) > 5 and save_path is not None:
        result_save_path = save_path + ransac_name
        _draw_and_save_matches(image0_origin, image1_origin, mkpts0, mkpts1, result_save_path)
    
    return Sen_pts, Ref_pts, mconf


def _draw_and_save_matches(img0, img1, kpts0, kpts1, save_path):

    try:
        kpts0_cv = [cv2.KeyPoint(kpts0[i][0], kpts0[i][1], 1) for i in range(len(kpts0))]
        kpts1_cv = [cv2.KeyPoint(kpts1[i][0], kpts1[i][1], 1) for i in range(len(kpts1))]
        
        if len(kpts0_cv) > 0:
            matches = [cv2.DMatch(idx, idx, 0) for idx in range(len(kpts0_cv))]
            img_matches = cv2.drawMatches(img0, kpts0_cv, img1, kpts1_cv, matches, None)
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, img_matches)
            print(f"Matches saved to {save_path}")
    except Exception as e:
        print(f"Warning: Failed to save matches visualization: {e}")
