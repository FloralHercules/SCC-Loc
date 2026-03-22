from PIL import Image
import cv2
import numpy as np
import os
from Matching_Models.RoMa.roma import roma_outdoor
# Remove skimage imports
# from skimage.measure import ransac
# from skimage.transform import ProjectiveTransform

def Roma_Init():
    # Create model
    device = 'cuda'
    root_path = './Matching_Models/RoMa/'
    model_path = root_path + "ckpt/roma_outdoor.pth"
    dinov2_path = root_path + 'ckpt/dinov2_vitl14_pretrain.pth'
    roma_model = roma_outdoor(device=device, weights=model_path, dinov2_weights=dinov2_path)
    return roma_model


def Roma_match(image0, image1, roma_model, save_path, ransac_name, need_ransac , show_matches):
    result_save_path = save_path + ransac_name
    image0_origin = image0
    image1_origin = image1
    device = 'cuda'
    W_A, H_A = image0.shape[1], image0.shape[0]
    W_B, H_B = image1.shape[1], image1.shape[0]
    image1_PIL = Image.fromarray(image0)
    image2_PIL = Image.fromarray(image1)
    
    warp, certainty = roma_model.match(image1_PIL, image2_PIL, device=device)
    
    matches, conf = roma_model.sample(warp, certainty, num=5000) 
    keypoints_left, keypoints_right = roma_model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)
    mkpts0, mkpts1 = keypoints_left.cpu().numpy(), keypoints_right.cpu().numpy()

    n_inliers1 = 0
    inliers = [None] # Default

    if len(mkpts0) > 8 and need_ransac:
        src_pts = np.float32(mkpts0).reshape(-1, 1, 2)
        dst_pts = np.float32(mkpts1).reshape(-1, 1, 2)
        
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.USAC_MAGSAC, 3.0, maxIters=5000, confidence=0.9999)
        
        if mask is not None:
             matchesMask = mask.ravel().tolist()
             inliers = [i for i, m in enumerate(matchesMask) if m == 1]
             inliers_bool = np.array(matchesMask, dtype=bool) 
             n_inliers1 = np.sum(inliers_bool)
             mkpts0 = mkpts0[inliers_bool]
             mkpts1 = mkpts1[inliers_bool]
             conf = conf[inliers_bool]
        else:
             mkpts0 = []
             mkpts1 = []
             conf = []
             
    inlier_keys_left = [[point[0], point[1]] for point in mkpts0]
    inlier_keys_right = [[point[0], point[1]] for point in mkpts1]
    conf = conf.tolist()
    if show_matches and len(inlier_keys_left) > 5:

        inlier_keypoints_left1 = [cv2.KeyPoint(point[0], point[1], 1) for point in mkpts0[inliers]]
        inlier_keypoints_right1 = [cv2.KeyPoint(point[0], point[1], 1) for point in mkpts1[inliers]]
        placeholder_matches = [cv2.DMatch(idx, idx, 1) for idx in range(n_inliers1)]

        image3 = cv2.drawMatches(image0_origin, inlier_keypoints_left1, image1_origin, inlier_keypoints_right1, placeholder_matches,
                                 None)
            
        
        save_dir = os.path.dirname(result_save_path)
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(result_save_path, image3)
        
    return inlier_keys_left, inlier_keys_right, conf


def draw_matches(im_A, kpts_A, im_B, kpts_B):
    kpts_A = [cv2.KeyPoint(x,y,1.) for x,y in kpts_A.cpu().numpy()]
    kpts_B = [cv2.KeyPoint(x,y,1.) for x,y in kpts_B.cpu().numpy()]
    matches_A_to_B = [cv2.DMatch(idx, idx, 0.) for idx in range(len(kpts_A))]
    im_A, im_B = np.array(im_A), np.array(im_B)
    ret = cv2.drawMatches(im_A, kpts_A, im_B, kpts_B,
                    matches_A_to_B, None)
    return ret
