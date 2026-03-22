import cv2
import numpy as np
import os
import sys
import torch
from types import SimpleNamespace
from PIL import Image
# 确保能导入同目录下的 load_model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from load_model import load_model

def MINIMA_Init(method_name='MINIMA_Roma'):
    """
    init the MINIMA model
    according the method_name parse the sub-method (roma, loftr, sp_lg, xoftr)
    """
    if '_' in method_name:
        sub_method = method_name.split('_')[1].lower()
    else:
        sub_method = 'roma' 

    if sub_method == 'lightglue': sub_method = 'sp_lg'
    
    print(f"Initializing MINIMA model: {sub_method}...")

    args = SimpleNamespace()
    args.method = sub_method
    
    weights_root = './Matching_Models/MINIMA/weights'
    
    args.ckpt = None
    args.ckpt2 = 'large' # For Roma
    args.thr = 0.2       # For LoFTR
    args.match_threshold = 0.3 # For XoFTR
    args.fine_threshold = 0.1  # For XoFTR

    if sub_method == 'roma':
        args.ckpt = os.path.join(weights_root, 'minima_roma.pth')
    elif sub_method == 'loftr':
        args.ckpt = os.path.join(weights_root, 'minima_loftr.ckpt')
    elif sub_method == 'sp_lg':
        args.ckpt = os.path.join(weights_root, 'minima_lightglue.pth')
    elif sub_method == 'xoftr':
        args.ckpt = os.path.join(weights_root, 'minima_xoftr.ckpt')

    try:
        matcher = load_model(sub_method, args, use_path=False, test_orginal_megadepth=False, return_object=True)
        return matcher
    except Exception as e:
        print(f"Error loading MINIMA model: {e}")
        print(f"Please check if weight files exist in {weights_root}")
        sys.exit(1)


def MINIMA_match(image0, image1, matcher_model, save_path, ransac_name, need_ransac=False, show_matches=False):

    roma_model = matcher_model.model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    H_A, W_A = image0.shape[:2]
    H_B, W_B = image1.shape[:2]

    img0_rgb = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB)
    img1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

    with torch.no_grad():
        if hasattr(roma_model, 'match'):
            image0_PIL = Image.fromarray(img0_rgb)
            image1_PIL = Image.fromarray(img1_rgb)
            
            warp, certainty = roma_model.match(image0_PIL, image1_PIL, device=device, batched=False)
            matches, mconf = roma_model.sample(warp, certainty, num=3000)
            kpts0, kpts1 = roma_model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)
            
            mkpts0 = kpts0.cpu().numpy()
            mkpts1 = kpts1.cpu().numpy()
            mconf = mconf.cpu().numpy()

        elif hasattr(matcher_model, 'from_cv_imgs'):

            match_res = matcher_model.from_cv_imgs(img0_rgb, img1_rgb)
            
            mkpts0 = match_res['mkpts0'] 
            mkpts1 = match_res['mkpts1']
            mconf = match_res['mconf']

        else:
            import torchvision.transforms.functional as TF
            t0 = TF.to_tensor(img0_rgb).unsqueeze(0).to(device)
            t1 = TF.to_tensor(img1_rgb).unsqueeze(0).to(device)
            
            output = roma_model(t0, t1)
            mkpts0, mkpts1, mconf = output['mkpts0'].cpu().numpy(), output['mkpts1'].cpu().numpy(), output['mconf'].cpu().numpy()

    conf_threshold = 0.00
    mask_conf = mconf > conf_threshold
    
    mkpts0 = mkpts0[mask_conf]
    mkpts1 = mkpts1[mask_conf]
    mconf = mconf[mask_conf]

    sort_indices = np.argsort(mconf)[::-1]
    mkpts0 = mkpts0[sort_indices]
    mkpts1 = mkpts1[sort_indices]
    mconf = mconf[sort_indices]

    if len(mkpts0) >= 4 and need_ransac:
        H, mask = cv2.findHomography(mkpts0, mkpts1, cv2.USAC_MAGSAC, 3.0, maxIters=2000, confidence=0.999)
        if mask is not None:
            mask = mask.ravel().astype(bool)
            mkpts0 = mkpts0[mask]
            mkpts1 = mkpts1[mask]
            mconf = mconf[mask]

    Sen_pts = mkpts0.tolist()
    Ref_pts = mkpts1.tolist()
    mconf = mconf.tolist()

    if show_matches and len(Sen_pts) > 0:
        result_save_path = os.path.join(save_path, ransac_name)
        os.makedirs(os.path.dirname(result_save_path), exist_ok=True)
        
        # 绘图前取 Top N 点，防止图太乱
        vis_top_n = min(len(Sen_pts), 100)
        kpts0_vis = [cv2.KeyPoint(p[0], p[1], 1) for p in Sen_pts[:vis_top_n]]
        kpts1_vis = [cv2.KeyPoint(p[0], p[1], 1) for p in Ref_pts[:vis_top_n]]
        matches_vis = [cv2.DMatch(i, i, 0) for i in range(vis_top_n)]
        
        vis_img = cv2.drawMatches(image0, kpts0_vis, image1, kpts1_vis, matches_vis, None, 
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite(result_save_path, vis_img)

    return Sen_pts, Ref_pts, mconf