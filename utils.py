from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # 解除像素限制
from Matching_Models.RoMa.demo.Roma_match1 import Roma_Init, Roma_match
# from Matching_Models.MINIMA.load_model import load_loftr, load_roma, load_sp_lg
from Matching_Models.MINIMA.MINIMA_match1 import MINIMA_Init, MINIMA_match
from Matching_Models.RoMaV2.demo.RoMaV2_match import RoMaV2_Init, RoMaV2_match
from pyproj import Transformer
import shutil
import glob
import pickle
from matplotlib import pyplot as plt
import matplotlib.cm as cm # 需要导入 matplotlib 的 colormap
from Retrieval_Models.multi_model_loader import get_Model
import math
import json
from Retrieval_Models.feature_extract import *
import os
import random

import time
import cv2
from math import sqrt
import numpy as np
from PIL import Image
import tifffile
from scipy.optimize import least_squares
# 在 utils.py 顶部添加
from scipy.spatial import Voronoi, Delaunay
from scipy.spatial.distance import cdist

# === [新增] 角度归一化 ===
def normalize_angle(angle):
    """normalize angle to [-180, 180) degrees"""
    angle = (angle + 180) % 360 - 180
    return angle

# === [新增] 计算详细的 6自由度 误差 ===
def calculate_6dof_errors(truePos, pred_BLH, pred_angle):
    """    
    calculate the horizontal error in meters, vertical error in meters, and attitude errors in degrees.
    """
    if pred_BLH['B'] is None or pred_angle is None:
        return None

    err_lat_m = (pred_BLH['B'] - truePos['lat']) * 111320 
    err_lon_m = (pred_BLH['L'] - truePos['lon']) * 111320 * np.cos(np.deg2rad(truePos['lat']))
    err_horz = np.sqrt(err_lat_m**2 + err_lon_m**2)
    
    err_alt = abs(pred_BLH['H'] - truePos['abs_height'])

    true_roll = truePos.get('roll', 0)
    true_pitch = truePos.get('pitch', 0)
    true_yaw = truePos.get('yaw', 0)

    err_roll = abs(normalize_angle(pred_angle[0] - true_roll))
    err_pitch = abs(normalize_angle(pred_angle[1] - true_pitch))
    err_yaw = abs(normalize_angle(pred_angle[2] - true_yaw))

    return {
        'err_horz': err_horz,
        'err_lat': err_lat_m,
        'err_lon': err_lon_m,
        'err_alt': err_alt,
        'err_roll': err_roll,
        'err_pitch': err_pitch,
        'err_yaw': err_yaw,
        'pred_lat': pred_BLH['B'],
        'pred_lon': pred_BLH['L'],
        'pred_alt': pred_BLH['H'],
        'pred_roll': pred_angle[0],
        'pred_pitch': pred_angle[1],
        'pred_yaw': pred_angle[2]
    }


def select_best_index_fusion(inliers_list, reproj_error_list, retrieval_score_list, 
                             uncertainty_list, BLH_list, 
                             voting_radius=20.0):
    """
    CD-RAPS Strategy
    select the best index based on a fusion of multiple criteria: inliers, reprojection error, uncertainty, and retrieval score, with spatial consistency voting.
    """
    n_candidates = len(inliers_list)
    if n_candidates == 0: return 0
    
    inliers = np.array(inliers_list, dtype=np.float32)
    errors = np.array(reproj_error_list, dtype=np.float32)
    
    unc_temp = []
    for u in uncertainty_list:
        if u is None: unc_temp.append(np.nan)
        else: unc_temp.append(u)
    uncertainties = np.array(unc_temp, dtype=np.float32)
    

    r_scores = np.array(retrieval_score_list, dtype=np.float32)

    valid_mask = np.isfinite(inliers) & np.isfinite(errors) & np.isfinite(uncertainties)
    
    if not np.any(valid_mask):
        return np.argmax(np.nan_to_num(inliers, nan=-1.0, posinf=-1.0, neginf=-1.0))

    def normalize_dim(data, mask, mode='positive'):
        mask = mask & (data > 0)
        valid_data = data[mask]
        valid_data = data[mask]
        
        if len(valid_data) == 0:
            return np.zeros_like(data)
            
        d_min = np.min(valid_data)
        d_max = np.max(valid_data)
        d_range = d_max - d_min
        
        norm_data = np.zeros_like(data)
        
        if d_range < 1e-6:

            norm_data[mask] = 1.0
        else:

            if mode == 'positive':
                norm_data[mask] = (data[mask] - d_min) / d_range
            else: 
                norm_data[mask] = 1.0 - (data[mask] - d_min) / d_range
                
        return norm_data


    norm_inliers = normalize_dim(inliers, valid_mask, mode='positive')
    norm_errors = normalize_dim(errors, valid_mask, mode='negative')
    norm_unc = normalize_dim(uncertainties, valid_mask, mode='negative')

    r_mask = np.isfinite(r_scores)
    norm_retrieval = normalize_dim(r_scores, r_mask, mode='positive')
    
    # base score fusion
    base_scores = (0.20 * norm_inliers) + \
                  (0.35 * norm_errors) + \
                  (0.35 * norm_unc) + \
                  (0.10 * norm_retrieval)
    

    final_scores = np.full(n_candidates, -1.0, dtype=np.float32)
    final_scores[valid_mask] = base_scores[valid_mask]


    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) <= 1:
        return np.argmax(final_scores)

    coords = []
    spatial_valid_indices = []
    
    for idx in valid_indices:
        b = BLH_list[idx].get('B')
        l = BLH_list[idx].get('L')
        if b is not None and l is not None:
            coords.append([b, l])
            spatial_valid_indices.append(idx)
        else:
            pass
            
    if not coords:
        return np.argmax(final_scores)

    coords = np.array(coords)
    spatial_valid_indices = np.array(spatial_valid_indices)
    
    mean_lat = coords[0, 0]
    lat_scale = 111320.0
    lon_scale = 111320.0 * np.cos(np.deg2rad(mean_lat))
    
    coords_m = coords.copy()
    coords_m[:, 0] = (coords[:, 0] - mean_lat) * lat_scale
    coords_m[:, 1] = (coords[:, 1] - coords[0, 1]) * lon_scale
    
    dist_mat = cdist(coords_m, coords_m) 
    
    n_spatial = len(spatial_valid_indices)
    vote_bonus = np.zeros(n_spatial)
    
    # vote score for each candidate based on its neighbors within voting_radius
    for i in range(n_spatial):

        idx_global = spatial_valid_indices[i]
        

        neighbor_mask = (dist_mat[i] < voting_radius) & (dist_mat[i] > 1e-3)
        neighbor_loc_idxs = np.where(neighbor_mask)[0]
        
        if len(neighbor_loc_idxs) == 0:
            continue
        
        current_bonus = 0.0
        vote_count = 0
        

        for n_loc_idx in neighbor_loc_idxs:

            idx_neighbor = spatial_valid_indices[n_loc_idx]
            dist = dist_mat[i, n_loc_idx]
            

            neighbor_quality = final_scores[idx_neighbor] 
            
            if neighbor_quality < 0.3: 
                continue

            weight_dist = 1.0 - (dist / voting_radius)
            
            vote_val = neighbor_quality * weight_dist
            
            current_bonus += vote_val
            vote_count += 1
            
            if vote_count >= 5: 
                break
        
        max_bonus = base_scores[idx_global] * 0.5
        vote_bonus[i] = min(current_bonus * 0.2, max_bonus)

    # final score
    for k, idx_global in enumerate(spatial_valid_indices):
        final_scores[idx_global] += vote_bonus[k]

    return np.argmax(final_scores)

# def select_best_index_fusion(inliers_list, reproj_error_list, retrieval_score_list, 
#                              uncertainty_list, BLH_list, 
#                              voting_radius=15.0):
#     """
#     Select the best index only based on the inlier count
#     """
#     n_candidates = len(inliers_list)
#     if n_candidates == 0: return 0
    
#     inliers = np.array(inliers_list, dtype=np.float32)
    
#     best_index = np.argmax(inliers)
    
#     return best_index

def angles_from_pnp_rvec_heading_pitch(rvec):
    """
    [SCI 改进版] 从 solvePnP 的 rvec 计算欧拉角
    
    改进点：
    不再单纯使用光轴(Z轴)计算Yaw，而是根据Pitch角度自适应选择参考向量。
    解决了无人机正射(俯视)拍摄时，Z轴垂直向下导致Yaw角计算奇异(Gimbal Lock)的问题。
    """
    # 1. 获取旋转矩阵 R_cw (World -> Camera)
    R_cw, _ = cv2.Rodrigues(rvec)
    # 2. 转为 R_wc (Camera -> World)
    R_wc = R_cw.T
    
    # --- A. 计算 Pitch (俯仰角) ---
    # 相机光轴 (+Z) 在世界系下的 Z 分量
    # Camera系: [0, 0, 1] -> World系
    cam_z_w = R_wc @ np.array([0, 0, 1]) 
    # Z分量向下为负，Pitch = asin(z_component)
    # 假设世界坐标系Z轴向上 (UTM习惯)
    pitch_rad = np.arcsin(np.clip(cam_z_w[2], -1.0, 1.0)) 
    pitch_deg = np.degrees(pitch_rad)
    
    # --- B. 计算 Yaw (航向角) ---
    # 关键点：当 Pitch 接近 -90 度(俯视)时，Z轴无法指示方向。
    # 此时使用"图像上方" (Camera -Y) 作为机头指向向量。
    
    if pitch_deg < -45: 
        # Case 1: 俯视/正射模式 (Pitch 接近 -90)
        # OpenCV相机系: Y轴向下，所以"图像上方"是 -Y = [0, -1, 0]
        # 这个向量在俯视时，水平指向机头方向
        ref_vec_c = np.array([0, -1, 0])
    else:
        # Case 2: 平视/斜视模式
        # 虽然此时 Z 轴也可以用，但为了保持与 Gimbal 逻辑一致(机头方向)，
        # 依然建议使用 -Y 轴 (假设云台Yaw跟随无人机Yaw)
        ref_vec_c = np.array([0, -1, 0]) 
        
        # 备注：如果你之前的代码平视时Yaw定义不同，可以取消下面这行的注释用回Z轴：
        # ref_vec_c = np.array([0, 0, 1])

    # 将参考向量转到世界系
    ref_vec_w = R_wc @ ref_vec_c
    
    # 提取东(X)北(Y)分量
    # 假设 World 系: X=East, Y=North (符合 UTM 定义)
    east = ref_vec_w[0]
    north = ref_vec_w[1]
    
    # 计算 Yaw (北偏东为正: atan2(East, North))
    yaw_deg = np.degrees(np.arctan2(east, north))
    
    # 强制 Roll = 0 (因为绝大多数无人机照片 Roll 都很小，强制置0可减少噪声)
    roll_deg = 0.0
    
    return np.array([roll_deg, pitch_deg, yaw_deg])

def img_name(sensing_path):
    file_name_with_ext = os.path.basename(sensing_path)
    file_name_without_ext, _ = os.path.splitext(file_name_with_ext)
    return file_name_without_ext


def load_data(save_path, *args):
    with open(save_path, 'rb') as file:
        data = pickle.load(file)
    results = tuple(data[arg] for arg in args if arg in data)

    return results

def save_data(filename, **kwargs):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    data = kwargs
    with open(filename, 'wb') as file:
        pickle.dump(data, file)


def get_jpg_files(folder_path):
    # Ensure the input path exists
    if not os.path.exists(folder_path):
        print("Path does not exist. Please check your input!")
        return []

    # If the path is a file rather than a folder, return an empty list
    if not os.path.isdir(folder_path):
        print("The input is a file path, not a folder path!")
        return []

    # Use the glob module to search for all JPG files, including uppercase and lowercase extensions
    jpg_files = glob.glob(os.path.join(folder_path, '*.JPG'))

    return jpg_files


def copy_image(src_path, dest_path, new_name):
    dest_file_path = os.path.join(dest_path, new_name)
    shutil.copy2(src_path, dest_file_path)


def view_center(region, config, truePos, initialX, initialY, ref_resolution, reverseMatRotation):
    '''This function estimates the location of the center of drone image (the center of view)'''
    pitch = -truePos['pitch'] / 180 * np.pi
    yaw = -truePos['yaw'] / 180 * np.pi
    utm_x, utm_y = deg2utm(region, config, truePos['lon'], truePos['lat'])
    delta_y = truePos['rel_height'] / np.tan(pitch) * np.cos(yaw)
    delta_x = -truePos['rel_height'] / np.tan(pitch) * np.sin(yaw)
    UTM_Y_c = utm_y + delta_y
    UTM_X_c = utm_x + delta_x
    x_center0 = int((UTM_X_c - initialX) / ref_resolution)
    y_center0 = int((initialY - UTM_Y_c) / ref_resolution)
    refCoordinate=np.array([x_center0, y_center0, 1])
    
    if reverseMatRotation is not None:
        refCoordinate2 = refCoordinate @ reverseMatRotation.T
        x_center, y_center = refCoordinate2[0], refCoordinate2[1]
    else:
        x_center, y_center = x_center0, y_center0
        
    return x_center, y_center

# ================= 新增裁剪核心功能 =================
def crop_geo_images_with_offset(region, config, opt, truePos, ref_map_full, dsm_map_full):
    """
    根据无人机位置，添加随机偏移，裁剪出4平方公里的搜索区域（2km x 2km）。
    同时更新用于后续计算的config参数（原点坐标等）。
    """
    Ref_type = opt.Ref_type
    
    # 1. 获取原始大图参数
    orig_initialX = config[f'{region}_{Ref_type}_REF_initialX']
    orig_initialY = config[f'{region}_{Ref_type}_REF_initialY']
    ref_res = config[f'{region}_{Ref_type}_REF_resolution']
    dsm_res = config[f'{region}_{Ref_type}_DSM_resolution']
    
    # 原始的对齐坐标 (Pixel coordinates used for alignment)
    orig_ref_coor = config[f'{region}_{Ref_type}_REF_COORDINATE']
    orig_dsm_coor = config[f'{region}_{Ref_type}_DSM_COORDINATE']

    # 2. 计算无人机当前的UTM坐标
    uav_utm_x, uav_utm_y = deg2utm(region, config, truePos['lon'], truePos['lat'])
    
    # 3. 设定裁剪参数
    crop_size_m = opt.crop_size_m  # 裁剪区域边长1000米 (1平方公里)
    half_size = crop_size_m / 2.0 
    
    # 4. 生成随机偏移 (模拟定位误差)
    # 偏移范围设定为 ±280米。
    offset_x = random.uniform(-120, 120) # [-280, 280]
    offset_y = random.uniform(-120, 120) # [-280, 280]
    
    # 5. 计算裁剪框中心的UTM坐标
    center_crop_x = uav_utm_x + offset_x
    center_crop_y = uav_utm_y + offset_y
    
    # 6. 计算裁剪框 左上角 (Top-Left) 的 UTM 坐标
    # 注意：UTM坐标系中，Y轴向上为正。但像素坐标系通常Y轴向下。
    tl_crop_utm_x = center_crop_x - half_size
    tl_crop_utm_y = center_crop_y + half_size 
    
    # 7. 计算裁剪框在 Ref Map (卫星图) 上的像素范围
    ref_col_start = int((tl_crop_utm_x - orig_initialX) / ref_res)
    ref_row_start = int((orig_initialY - tl_crop_utm_y) / ref_res)
    
    ref_w_pixel = int(crop_size_m / ref_res)
    ref_h_pixel = int(crop_size_m / ref_res)
    
    ref_col_end = ref_col_start + ref_w_pixel
    ref_row_end = ref_row_start + ref_h_pixel
    
    # 边界检查 (Ref Map)
    h_ref, w_ref = ref_map_full.shape[:2]
    ref_col_start = max(0, ref_col_start)
    ref_row_start = max(0, ref_row_start)
    ref_col_end = min(w_ref, ref_col_end)
    ref_row_end = min(h_ref, ref_row_end)
    
    # 执行裁剪 Ref Map
    ref_crop = ref_map_full[ref_row_start:ref_row_end, ref_col_start:ref_col_end]
    
    # 8. 计算裁剪框在 DSM Map 上的像素范围
    # 既然代码用 match2pos 里的 dsm_offset 修正，我们按照比例裁剪即可。
    scale_factor = ref_res / dsm_res
    
    dsm_col_start = int(orig_dsm_coor[0] + (ref_col_start - orig_ref_coor[0]) * scale_factor)
    dsm_row_start = int(orig_dsm_coor[1] + (ref_row_start - orig_ref_coor[1]) * scale_factor)
    
    dsm_w_pixel = int(ref_w_pixel * scale_factor)
    dsm_h_pixel = int(ref_h_pixel * scale_factor)
    
    dsm_col_end = dsm_col_start + dsm_w_pixel
    dsm_row_end = dsm_row_start + dsm_h_pixel
    
    # 边界检查 (DSM Map)
    h_dsm, w_dsm = dsm_map_full.shape[:2]
    dsm_col_start = max(0, dsm_col_start)
    dsm_row_start = max(0, dsm_row_start)
    dsm_col_end = min(w_dsm, dsm_col_end)
    dsm_row_end = min(h_dsm, dsm_row_end)
    
    dsm_crop = dsm_map_full[dsm_row_start:dsm_row_end, dsm_col_start:dsm_col_end]

    # 9. 更新 Config 参数 (至关重要!)
    new_config = config.copy()
    
    # 更新 Ref 的起始 UTM 坐标
    new_ref_initialX = orig_initialX + ref_col_start * ref_res
    new_ref_initialY = orig_initialY - ref_row_start * ref_res 
    
    new_config[f'{region}_{Ref_type}_REF_initialX'] = new_ref_initialX
    new_config[f'{region}_{Ref_type}_REF_initialY'] = new_ref_initialY
    
    # 更新对齐坐标 (COORDINATE) 同名点在锚框外面，因此为负数
    new_ref_coor = [orig_ref_coor[0] - ref_col_start, orig_ref_coor[1] - ref_row_start]
    new_dsm_coor = [orig_dsm_coor[0] - dsm_col_start, orig_dsm_coor[1] - dsm_row_start]
    
    new_config[f'{region}_{Ref_type}_REF_COORDINATE'] = new_ref_coor
    new_config[f'{region}_{Ref_type}_DSM_COORDINATE'] = new_dsm_coor
    
    return ref_crop, dsm_crop, new_config

def filter_matches_spatial_grid(pts_uav, pts_ref, confidences, img_shape, grid_size=8, base_quota=3):
    """
    [SCI 创新版] 基于纹理分布的自适应网格过滤 (Texture-Aware Adaptive Spatial Filtering)
    
    创新点：不强制每个网格保留固定数量的点，而是根据该网格内的点密度动态分配配额。
    特征丰富的区域自动获得更多保留名额，特征稀疏区域自动减少。
    """
    if len(pts_uav) == 0:
        return pts_uav, pts_ref, confidences

    H, W = img_shape[:2]
    cell_h = H / grid_size
    cell_w = W / grid_size
    
    # 1. 将点分配到网格中
    grid_buckets = {} # key: (r, c), value: list of (index, confidence)
    
    for i, (u, v) in enumerate(pts_uav):
        c = int(u / cell_w)
        r = int(v / cell_h)
        r = min(max(r, 0), grid_size - 1)
        c = min(max(c, 0), grid_size - 1)
        
        if (r, c) not in grid_buckets:
            grid_buckets[(r, c)] = []
        grid_buckets[(r, c)].append((i, confidences[i]))

    keep_indices = []
    
    # 2. [自适应策略] 计算动态配额
    # 简单的策略：如果一个格子里原本匹配到的点非常多，说明这里纹理好，应该多留点。
    # 如果原本就只有 1-2 个点，说明这里大概率是弱纹理，应该少留。
    
    for (r, c), items in grid_buckets.items():
        # 按置信度降序排列
        items.sort(key=lambda x: x[1], reverse=True)
        
        n_candidates = len(items)
        
        # 动态配额公式：
        # 基础配额 + 奖励配额 (对数增长，防止聚堆)
        # 例子：如果格子里有 50 个候选点，extra ≈ log2(50) ≈ 5，总共留 3+5=8 个
        # 如果格子里只有 2 个候选点，extra ≈ 1，总共留 2 个（取最小值）
        extra_quota = int(np.log2(n_candidates + 1))
        dynamic_limit = base_quota + extra_quota
        
        # 还要设置一个绝对上限，防止某个点太密集
        dynamic_limit = min(dynamic_limit, base_quota * 3) 
        
        # 取前 dynamic_limit 个
        selected = items[:dynamic_limit]
        for idx, _ in selected:
            keep_indices.append(idx)
            
    keep_indices = np.array(keep_indices)
    
    return pts_uav[keep_indices], pts_ref[keep_indices], confidences[keep_indices]
    

# =========================================================================
# [核心函数 1] 计算局部标准差图 (保持不变，速度快)
# =========================================================================
def compute_local_saliency_map(img, w_size=15):
    """
    计算局部纹理显著性图 (Local Standard Deviation)。
    Args:
        img: 输入图像 (BGR 或 Gray)
        w_size: 局部窗口大小
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    img_f = gray.astype(np.float32) / 255.0
    
    # 利用积分图原理计算局部标准差: Var(X) = E[X^2] - (E[X])^2
    mean = cv2.boxFilter(img_f, -1, (w_size, w_size), normalize=True)
    sqr_mean = cv2.boxFilter(img_f**2, -1, (w_size, w_size), normalize=True)
    
    variance = np.abs(sqr_mean - mean**2)
    std_dev = np.sqrt(variance)
    
    # 归一化到 0-1
    saliency_map = cv2.normalize(std_dev, None, 0, 1, cv2.NORM_MINMAX)
    return saliency_map

# =========================================================================
# [核心函数 2] 带回退机制的筛选器
# =========================================================================
def filter_by_joint_saliency(uav_img, ref_img, pts_uav, pts_ref, confidences, 
                             alpha=0.5, min_samples=15):
    """
    [SCI 改进版 + 回退保护] 
    基于局部自适应纹理的筛选，若剩余点过少，则触发回退机制。
    
    Args:
        min_samples: 最少保留点数。如果严格筛选后少于此数，触发回退。
    """
    if len(pts_uav) == 0:
        return pts_uav, pts_ref, confidences

    # 1. 计算局部显著性图
    s_map_u = compute_local_saliency_map(uav_img, w_size=15)
    s_map_r = compute_local_saliency_map(ref_img, w_size=15)
    
    h_u, w_u = s_map_u.shape
    h_r, w_r = s_map_r.shape
    
    new_confidences = []
    strict_indices = [] # 严格模式下保留的索引
    all_joint_scores = [] # 记录所有点的纹理分，供回退使用
    
    # 2. 计算基底噪声水平 (用于严格筛选)
    noise_floor_u = np.mean(s_map_u) * 0.5 
    noise_floor_r = np.mean(s_map_r) * 0.5

    for i in range(len(pts_uav)):
        xu, yu = int(pts_uav[i][0]), int(pts_uav[i][1])
        xr, yr = int(pts_ref[i][0]), int(pts_ref[i][1])
        
        # 边界保护
        xu, yu = np.clip(xu, 0, w_u - 1), np.clip(yu, 0, h_u - 1)
        xr, yr = np.clip(xr, 0, w_r - 1), np.clip(yr, 0, h_r - 1)
        
        val_u = s_map_u[yu, xu]
        val_r = s_map_r[yr, xr]
        
        # 计算联合纹理分 (Joint Texture Score)
        # joint_score = np.sqrt(val_u * val_r)
        joint_score = max(val_u, val_r) * 0.8 + min(val_u, val_r) * 0.2
        all_joint_scores.append(joint_score)
        
        # 融合后的新置信度 (原始分 * 纹理加成)
        boosted_conf = confidences[i] * (1 + alpha * joint_score)
        # new_confidences.append(boosted_conf)
        new_confidences.append(confidences[i])
        
        # === 严格筛选逻辑 ===
        # 只有显著高于底噪才通过
        if val_u > noise_floor_u and val_r > noise_floor_r:
            strict_indices.append(i)

    new_confidences = np.array(new_confidences)
    strict_indices = np.array(strict_indices)
    
    # =============================================================
    # [核心逻辑] 回退检查 (Fallback Check)
    # =============================================================
    
    # 情况 A: 严格筛选后的点足够多 -> 使用严格筛选结果
    # if len(strict_indices) >= min_samples:
    #     return pts_uav[strict_indices], pts_ref[strict_indices], new_confidences[strict_indices]
    
    # # 情况 B: 点不够 -> 触发回退机制 (Soft Fallback)
    # else:
    #     # print(f"[Warning] Saliency filter too strict ({len(strict_indices)} left). Triggering Fallback...")
        
    #     # 策略：不要直接丢弃，而是根据 (原始置信度 * 纹理分) 进行综合排序
    #     # 强制保留前 K 个点 (K = min_samples * 1.5, 稍微多留点给 PnP RANSAC 筛)
    #     fallback_k = min(len(pts_uav), int(min_samples * 2))
        
    #     # 获取排序索引 (从大到小)
    #     sorted_indices = np.argsort(new_confidences)[::-1]
    #     top_k_indices = sorted_indices[:fallback_k]
    #     print(1)
    #     return pts_uav[top_k_indices], pts_ref[top_k_indices], new_confidences[top_k_indices]
    if len(strict_indices) == 0:
        # 如果严格筛选后没有点，回退到保留所有点
        return pts_uav, pts_ref, new_confidences
    
    return pts_uav[strict_indices], pts_ref[strict_indices], new_confidences[strict_indices]

# =========================================================================
# [创新核心 1] 基于 Voronoi 的自适应密度均衡 (Adaptive Density Equalization)
# =========================================================================
def filter_matches_voronoi_density(pts_uav, pts_ref, confidences, img_shape, keep_ratio=0.75):
    """
    利用 Voronoi 图的单元面积来判断特征点的局部密度。
    面积越小 -> 密度越大 -> 可能是树叶/噪点聚类 -> 降权或剔除。
    """
    n_points = len(pts_uav)
    if n_points < 10:
        return pts_uav, pts_ref, confidences

    # 1. 辅助函数：计算多边形面积
    def polygon_area(corners):
        n = len(corners)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += corners[i][0] * corners[j][1]
            area -= corners[j][0] * corners[i][1]
        return abs(area) / 2.0

    # 2. 构建 Voronoi 图
    # 为了防止边界点产生无限大的区域，我们在图像四周添加 4 个虚拟辅助点
    h, w = img_shape[:2]
    boundary_points = np.array([[-w, -h], [2*w, -h], [2*w, 2*h], [-w, 2*h]])
    points_all = np.vstack([pts_uav, boundary_points])
    
    try:
        vor = Voronoi(points_all)
    except Exception:
        # 共线或退化情况
        return pts_uav, pts_ref, confidences

    # 3. 计算每个特征点的 Voronoi 单元面积
    areas = []
    # 前 n_points 个点是我们关心的真实点
    for i in range(n_points):
        region_index = vor.point_region[i]
        region_indices = vor.regions[region_index]
        
        # 如果包含 -1，说明是无限区域（通常在边界），给一个较大的默认值
        if -1 in region_indices or len(region_indices) == 0:
            areas.append(np.inf)
        else:
            vertices = vor.vertices[region_indices]
            areas.append(polygon_area(vertices))
    
    areas = np.array(areas)
    
    # 4. 密度均衡策略
    # 将 Inf 替换为当前最大的有限面积，以免干扰统计
    finite_areas = areas[np.isfinite(areas)]
    if len(finite_areas) == 0: return pts_uav, pts_ref, confidences
    
    max_area = np.max(finite_areas)
    areas[np.isinf(areas)] = max_area * 1.1

    # 策略 A：概率保留 (Soft) - 面积越小(越挤)，保留概率越低
    # 这种方法比较优雅，不是硬阈值
    # median_area = np.median(areas)
    # prob = 1.0 - np.exp(-areas / (median_area * 0.5)) # Sigmoid-like
    # mask = np.random.rand(n_points) < prob
    
    # 策略 B：硬阈值剔除 (Hard) - 直接砍掉最挤的 Top X%
    # 更加稳定，适合工程复现
    threshold = np.percentile(areas, (1 - keep_ratio) * 100) # 比如丢弃面积最小的 25%
    
    # [创新点结合]：不仅看密度，结合置信度。
    # 如果一个点很挤(area小)，但置信度(confidence)极高，也可以网开一面。
    # Score = Norm(Area) + Norm(Confidence)
    norm_area = areas / max_area
    norm_conf = confidences / (np.max(confidences) + 1e-6)
    
    # 综合得分：密度分布权重 0.6，原始置信度 0.4
    joint_score = 0.6 * norm_area + 0.4 * norm_conf
    
    # 取前 keep_ratio 的点
    k = int(n_points * keep_ratio)
    top_k_idx = np.argsort(joint_score)[::-1][:k]
    
    return pts_uav[top_k_idx], pts_ref[top_k_idx], confidences[top_k_idx]


# =========================================================================
# [创新核心 2] 基于 Delaunay 的拓扑一致性过滤 (Topo-Filter)
# =========================================================================
def filter_matches_delaunay_topology(pts_uav, pts_ref, confidences, area_ratio_threshold=0.3):
    """
    Topo-Filter: 利用 Delaunay 三角剖分检查拓扑一致性。
    假设：无人机图和卫星图之间近似为相似变换（旋转+缩放）。
    对于正确的匹配点，其构成的三角形在两张图上的【面积比】应该接近全局尺度的平方，是一个常数。
    如果某个三角形的面积比异常，说明其顶点中有误匹配。
    """
    n_points = len(pts_uav)
    if n_points < 6: # 点太少构不成足够的三角形
        return pts_uav, pts_ref, confidences

    # 1. 在 UAV 图像上构建 Delaunay 三角网
    try:
        tri = Delaunay(pts_uav)
    except Exception:
        return pts_uav, pts_ref, confidences
        
    simplices = tri.simplices # (N_tri, 3) 索引

    # 2. 向量化计算三角形面积
    def calc_tri_areas(pts, indices):
        # 顶点坐标
        p1 = pts[indices[:, 0]]
        p2 = pts[indices[:, 1]]
        p3 = pts[indices[:, 2]]
        # 叉乘计算面积: 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
        return 0.5 * np.abs(p1[:,0]*(p2[:,1]-p3[:,1]) + 
                            p2[:,0]*(p3[:,1]-p1[:,1]) + 
                            p3[:,0]*(p1[:,1]-p2[:,1]))

    area_uav = calc_tri_areas(pts_uav, simplices)
    area_ref = calc_tri_areas(pts_ref, simplices) # 使用相同的索引在 ref 上计算

    # 3. 计算比例一致性
    # 避免除以 0
    valid_mask = (area_uav > 1e-4) & (area_ref > 1e-4)
    if np.sum(valid_mask) == 0:
        return pts_uav, pts_ref, confidences

    ratios = area_ref[valid_mask] / area_uav[valid_mask]
    
    # 核心假设：正确匹配的三角形，其面积缩放比例应该聚集在真实值附近
    # 使用中位数作为 Robust Estimate
    median_ratio = np.median(ratios)
    
    # 计算每个三角形的偏差程度
    ratio_diff = np.abs(ratios - median_ratio) / median_ratio
    
    # 4. 投票剔除机制 (Voting Mechanism)
    # 如果一个三角形是“坏”的（比例异常），它的三个顶点都投一票“坏票”。
    # 如果一个点积累了太多的坏票，它就是误匹配。
    
    point_bad_votes = np.zeros(n_points)
    point_total_votes = np.zeros(n_points)
    
    # 找出异常三角形的索引 (在 valid_mask 中的索引)
    bad_tri_local_indices = np.where(ratio_diff > area_ratio_threshold)[0]
    
    # 映射回原始 simplices 的索引
    valid_indices = np.where(valid_mask)[0]
    bad_tri_indices = valid_indices[bad_tri_local_indices]
    
    # 统计投票
    # 所有参与了三角形构建的点，分母+1
    for tri_idx in valid_indices:
        pts_idx = simplices[tri_idx]
        point_total_votes[pts_idx] += 1
        
    # 只有异常三角形的点，分子+1
    for tri_idx in bad_tri_indices:
        pts_idx = simplices[tri_idx]
        point_bad_votes[pts_idx] += 1
        
    # 5. 计算坏点率
    # 避免分母为0
    with np.errstate(divide='ignore', invalid='ignore'):
        bad_rate = point_bad_votes / (point_total_votes + 1e-6)
        
    # 剔除坏点率超过 50% 的点 (即它参与的三角形大部分都变形了)
    # 同时保留那些没参与三角剖分的孤立点（虽然Delaunay通常覆盖凸包内所有点，但为了安全）
    keep_mask = (bad_rate < 0.5) | (point_total_votes == 0)
    
    # 兜底：如果剔除太多，适当放宽
    if np.sum(keep_mask) < 8:
        # 回退策略：只剔除最差的 top 10%
        top_bad_indices = np.argsort(bad_rate)[::-1][:int(n_points*0.1)]
        keep_mask = np.ones(n_points, dtype=bool)
        keep_mask[top_bad_indices] = False
        
    return pts_uav[keep_mask], pts_ref[keep_mask], confidences[keep_mask]

def filter_matches_geometric_consistency(pts_uav, pts_ref, confidences, tolerance_scale=0.3, tolerance_angle=15):
    """
    [几何一致性过滤 - 增强版]
    同时检查：
    1. 缩放一致性 (Scale Consistency): 点对距离比值是否一致
    2. 方向一致性 (Direction Consistency): 向量角度是否一致 [新增]
    
    Args:
        tolerance_scale: 缩放容差 (0.3 表示允许 ±30% 误差)
        tolerance_angle: 角度容差 (度)，建议 10~15 度
    """
    n = len(pts_uav)
    if n < 4:
        return pts_uav, pts_ref, confidences

    # 1. 计算几何中心 (Centroid)
    # 使用加权中心可能更准，这里用几何中心即可
    center_uav = np.mean(pts_uav, axis=0)
    center_ref = np.mean(pts_ref, axis=0)
    
    # 2. 计算每个点相对于中心的向量 (Vector)
    vec_uav = pts_uav - center_uav # (N, 2)
    vec_ref = pts_ref - center_ref # (N, 2)
    
    # --- A. 缩放检查 (Scale Check) ---
    dist_uav = np.linalg.norm(vec_uav, axis=1) + 1e-6
    dist_ref = np.linalg.norm(vec_ref, axis=1) + 1e-6
    
    ratios = dist_ref / dist_uav
    median_ratio = np.median(ratios)
    
    scale_mask = (ratios >= median_ratio * (1 - tolerance_scale)) & \
                 (ratios <= median_ratio * (1 + tolerance_scale))

    # --- B. 方向检查 (Direction Check) ---
    # 计算每个向量的角度 (弧度 -pi ~ pi)
    ang_uav = np.arctan2(vec_uav[:, 1], vec_uav[:, 0])
    ang_ref = np.arctan2(vec_ref[:, 1], vec_ref[:, 0])
    
    # 计算角度差 (Ref - UAV)
    # 注意：如果两图有旋转，这个差值应该是一个常数 (Rotation Angle)
    # 我们取中位数作为估计的全局旋转角
    delta_ang = ang_ref - ang_uav
    
    # 角度归一化到 -pi ~ pi
    delta_ang = (delta_ang + np.pi) % (2 * np.pi) - np.pi
    
    # 估计全局旋转偏差 (Global Rotation Bias)
    global_rot = np.median(delta_ang)
    
    # 计算每个点的角度偏差
    ang_err = np.abs(delta_ang - global_rot)
    # 再次归一化 (处理跨越 180/-180 度的情况)
    ang_err = (ang_err + np.pi) % (2 * np.pi) - np.pi
    ang_err = np.abs(ang_err)
    
    # 转换为角度制比较
    limit_rad = np.deg2rad(tolerance_angle)
    dir_mask = ang_err < limit_rad

    # 3. 综合掩码 (同时满足缩放和方向约束)
    # 注意：距离太近的点方向计算不稳定，直接视为通过方向检查
    dist_threshold = 10.0 # 像素
    valid_dist = dist_uav > dist_threshold
    
    # 对于距离中心太近的点，不做方向检查（避免噪点），只看缩放
    final_mask = scale_mask & (dir_mask | (~valid_dist))
    
    return pts_uav[final_mask], pts_ref[final_mask], confidences[final_mask]


def Match2Pos_all(opt, region, config, uav_img0, finescale, K, ref_image, dsm_image, refLocY_list, refLocX_list, 
                  cut_H_list, cut_W_list, # <--- 变更为列表
                  save_path, method_dict, matRotation, retrieval_scores=None, prior_pitch=None):

    '''This function first matches drone image and the reference images 
    and then use the dsm data to solve pnp problem to get the drone position.'''
    Model_name = method_dict['matching_method']
    matching_model = method_dict['matching_model']

    if opt.pose_priori == 'yp':
        reverseMatRotation = cv2.invertAffineTransform(matRotation)
    else:
        reverseMatRotation = None
    resize_ratio = opt.resize_ratio

    # ... (参数获取逻辑不变) ...
    Ref_type = opt.Ref_type
    initialX = config[f'{region}_{Ref_type}_REF_initialX']
    initialY = config[f'{region}_{Ref_type}_REF_initialY']
    ref_resolution = config[f'{region}_{Ref_type}_REF_resolution']
    dsm_resolution = config[f'{region}_{Ref_type}_DSM_resolution']
    
    dsm_coor = config[f'{region}_{Ref_type}_DSM_COORDINATE']
    ref_coor = config[f'{region}_{Ref_type}_REF_COORDINATE']
    dsm_ratio = dsm_resolution / ref_resolution
    dsm_coor_scaled = [dsm_coor[0] * dsm_ratio, dsm_coor[1] * dsm_ratio]
    dsm_offset = (int(dsm_coor_scaled[0] - ref_coor[0]), int(dsm_coor_scaled[1] - ref_coor[1]))

    if resize_ratio<1:
        uav_img = cv2.resize(uav_img0, None, fx=resize_ratio, fy=resize_ratio)
    else:
        uav_img = uav_img0

    BLH_list = []
    inliers_list = []
    reproj_error_list = [] # [新增]
    angle_list = []        # [新增]
    retrieval_score_subset = [] # [新增]
    time_list = []
    uncertainty_list = []
    
    if not isinstance(refLocY_list, list):
        refLocX_list = [refLocX_list]
        refLocY_list = [refLocY_list]
        
    # 处理 retrieval_scores，确保长度匹配
    if retrieval_scores is not None:
         # 转换为 list 或 numpy
         if isinstance(retrieval_scores, np.ndarray):
             scores_list = retrieval_scores.flatten().tolist()
         else:
             scores_list = list(retrieval_scores)
    else:
         scores_list = [0] * len(refLocY_list)

    top_n = min(method_dict['retrieval_topn'], len(refLocY_list))

    for index in range(top_n):
        ransac_name = '/top{}_ransac.png'.format(index+1)
        refLocY = refLocY_list[index]
        refLocX = refLocX_list[index]
        current_score = scores_list[index] if index < len(scores_list) else 0 # 获取当前分数
        

        curr_cut_H = cut_H_list[index] if isinstance(cut_H_list, list) else cut_H_list
        curr_cut_W = cut_W_list[index] if isinstance(cut_W_list, list) else cut_W_list

        # 边界保护 (使用动态尺寸)
        if refLocX + curr_cut_W > ref_image.shape[0] or refLocY + curr_cut_H > ref_image.shape[1]:
            continue

        # 裁切 (使用动态尺寸)
        fineRef = ref_image[refLocX:refLocX+curr_cut_W, refLocY:refLocY+curr_cut_H]
        fineRef = cv2.resize(fineRef, None, fx=resize_ratio/finescale, fy=resize_ratio/finescale)

        match_time_start = time.time()
        # 调用匹配模型
        if Model_name == 'Roma':
            Sen_pts, Ref_pts, mconf = Roma_match(uav_img, fineRef, matching_model, save_path, ransac_name, need_ransac=True, show_matches=False)
        elif Model_name == 'RoMaV2':
            Sen_pts, Ref_pts, mconf = RoMaV2_match(uav_img, fineRef, matching_model, save_path, ransac_name, need_ransac=True, show_matches=False)
        elif 'MINIMA' in Model_name:
            Sen_pts, Ref_pts, mconf = MINIMA_match(uav_img, fineRef, matching_model, save_path, ransac_name, need_ransac=True, show_matches=False) 
        else:
            print(f'Method name {Model_name} is wrong!!!')

        match_time_end = time.time()
        single_match_time = match_time_end-match_time_start
       
        pts_uav = np.array(Sen_pts)
        pts_ref = np.array(Ref_pts)
        mconf = np.array(mconf)
        
        """C-SATSF Mechanism"""
        #------------------------------------------------------------#
        # Spatial Equalization
        if len(pts_uav) > 50:
            pts_uav, pts_ref, mconf = filter_matches_spatial_grid(pts_uav, pts_ref, mconf, img_shape=uav_img.shape, grid_size=8) # 10
        # Texture Verification
        if len(pts_uav) > 40:
            pts_uav, pts_ref, mconf = filter_by_joint_saliency(
                uav_img, fineRef, pts_uav, pts_ref, mconf, 
                alpha=0.5,       # Texture bonus weight
                min_samples=15   # backup threshold for fallback (if too strict)
            )
        # Structure-Consistent Refinement
        ## Topologic
        if len(pts_uav) > 30:
            pts_uav, pts_ref, mconf = filter_matches_delaunay_topology(
                pts_uav, pts_ref, mconf, 
                area_ratio_threshold=0.4 # allow 40% area ratio deviation
            )
        ## Geometric
        if len(pts_uav) > 8:
            pts_uav, pts_ref, mconf = filter_matches_geometric_consistency(
                    pts_uav, pts_ref, mconf, 
                    tolerance_scale=0.3, 
                    tolerance_angle=20
            )
        #------------------------------------------------------------#
        
        if len(pts_uav) > 8:
            src_pts = np.float32(pts_uav).reshape(-1, 1, 2)
            dst_pts = np.float32(pts_ref).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.USAC_MAGSAC, 3.0, maxIters=5000, confidence=0.9999)
            if mask is not None:
                inliers_bool = mask.ravel().astype(bool)
                pts_uav = pts_uav[inliers_bool]
                pts_ref = pts_ref[inliers_bool]
            else:
                pts_uav, pts_ref = np.empty((0, 2)), np.empty((0, 2))
        else:
            pts_uav, pts_ref = np.empty((0, 2)), np.empty((0, 2))


        Sen_pts = pts_uav.tolist()
        Ref_pts = pts_ref.tolist()
        reproj_error = float('inf') 
        pose_unc = float('inf') 
        cam_angle = None
        inliers_count = 0

        # Ready to solve PnP if we have enough matches after filtering
        if len(Ref_pts)>=5:
            refCoordinate = np.array(Ref_pts)/resize_ratio*finescale + np.array([refLocY, refLocX])
            if opt.pose_priori == 'yp':
                refCoordinate1 = np.hstack([refCoordinate, np.ones((refCoordinate.shape[0], 1))])
                refCoordinate = refCoordinate1 @ reverseMatRotation.T
                UTM_X = refCoordinate[:, 0] * ref_resolution + initialX
                UTM_Y = initialY - refCoordinate[:, 1] * ref_resolution
            else:
                UTM_X = refCoordinate[:, 0] * ref_resolution + initialX
                UTM_Y = initialY - refCoordinate[:, 1] * ref_resolution

            dsm_x = refCoordinate[:, 1] + dsm_offset[1]
            dsm_y = refCoordinate[:, 0] + dsm_offset[0]
            dsm_x, dsm_y = (dsm_x + 1) / dsm_ratio - 1, (dsm_y + 1) / dsm_ratio - 1
            dsm_x1 = np.clip(dsm_x.astype(int), 0, dsm_image.shape[0] - 1)
            dsm_y1 = np.clip(dsm_y.astype(int), 0, dsm_image.shape[1] - 1)
            DSM = dsm_image[dsm_x1, dsm_y1]

            match_points = np.array(Sen_pts)/resize_ratio
            
            BLH, cam_angle, inliers_count, inliers_indices, reproj_error, pose_unc = estimate_drone_pose(
                region, config, match_points, K, UTM_X, UTM_Y, DSM, 
                prior_pitch= prior_pitch
            )         
            if config['SHOW_RETRIEVAL_RESULT']:
                process_and_save_matches(Sen_pts, Ref_pts, inliers_indices, uav_img, fineRef, save_path+f'/{index+1}-{inliers_count}.png')
        else:
            BLH = {'B': None, 'L': None, 'H': None}
            inliers_count = 0

        PnP_time_end = time.time()
        single_PnP_time = PnP_time_end - match_time_end
        time_list.append([single_match_time, single_PnP_time])
        
        
        BLH_list.append(BLH)
        inliers_list.append(inliers_count)
        reproj_error_list.append(reproj_error) 
        angle_list.append(cam_angle)           
        retrieval_score_subset.append(current_score)
        uncertainty_list.append(pose_unc)

    match_time = [t[0] for t in time_list]
    pnp_time = [t[1] for t in time_list]
    
    return BLH_list, inliers_list, match_time, pnp_time, reproj_error_list, retrieval_score_subset, angle_list, uncertainty_list

def process_and_save_matches(Sen_pts, Ref_pts, inliers, uav_image, ref_image, save_path):
    if len(inliers) > 0:
        if isinstance(inliers[0], (list, np.ndarray, tuple)):
            Sen_pts_temp = [Sen_pts[i[0]] for i in inliers]
            Ref_pts_temp = [Ref_pts[i[0]] for i in inliers]
        else:
            Sen_pts_temp = [Sen_pts[i] for i in inliers]
            Ref_pts_temp = [Ref_pts[i] for i in inliers]
    else:
        Sen_pts_temp = []
        Ref_pts_temp = []

    h1, w1 = uav_image.shape[:2]
    h2, w2 = ref_image.shape[:2]
    
    height = max(h1, h2)
    width = w1 + w2
    vis_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    vis_img[:h1, :w1, :] = uav_image
    vis_img[:h2, w1:w1+w2, :] = ref_image

    for pt1, pt2 in zip(Sen_pts_temp, Ref_pts_temp):
        pt1 = (int(pt1[0]), int(pt1[1]))
        pt2 = (int(pt2[0] + w1), int(pt2[1])) 
        
        cv2.line(vis_img, pt1, pt2, (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
        
        cv2.circle(vis_img, pt1, 2, (0, 0, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(vis_img, pt2, 2, (0, 0, 255), -1, lineType=cv2.LINE_AA)

    cv2.imwrite(save_path, vis_img)

def pos2error(truePos, BLH_list, inliers_list, reproj_error_list, retrieval_score_subset, angle_list, uncertainty_list):
    """
    calculate the error between the predicted BLH and the true position, and select the best prediction based on inliers, reprojection error, retrieval score, and uncertainty.
    """
    if len(inliers_list) == 0:
        return None, 10000, None

    # select optimal position index
    max_index = select_best_index_fusion(inliers_list, reproj_error_list, retrieval_score_subset, uncertainty_list, BLH_list)
    
    best_BLH = BLH_list[max_index]
    best_angle = angle_list[max_index]
    
    # cal 6dof error
    detailed_error = calculate_6dof_errors(truePos, best_BLH, best_angle)
    

    if detailed_error is None:
        return None, 10000, None
        
    return detailed_error, detailed_error['err_horz'], detailed_error['pred_lon'] 


def computeCameraMatrix(truePos):
    """compute intrinsic camera matrix K from truePos parameters"""
    imW0 = truePos['width']
    imH0 = truePos['height']
    cameraSize = truePos['cam_size']
    focal = truePos['focal_len']
    pixelSize_x = cameraSize / np.sqrt(imW0**2 + imH0**2)
    pixelSize_y = cameraSize / np.sqrt(imW0**2 + imH0**2)
    focalx = focal / pixelSize_x
    focaly = focal / pixelSize_y
    centerX = imW0 / 2
    centerY = imH0 / 2
    K = np.array([
        [focalx, 0, centerX],
        [0, focaly, centerY],
        [0, 0, 1]
    ])
    return K

def estimate_drone_pose(region, config, match_points, K, UTM_X, UTM_Y, DSM, dist_coeffs=None, prior_pitch=None):
    """Estimate the drone pose using the estimated camera matrix and match points with prior information"""
    pose_3d = np.column_stack((UTM_X, UTM_Y, DSM))
    if dist_coeffs is None: dist_coeffs = np.zeros((5, 1))
    
    reproj_error = float('inf')
    pose_uncertainty = float('inf') 
    if len(pose_3d) < 4:
        return {'B': None, 'L': None, 'H': None}, None, 0, [], reproj_error, float('inf')
        
    try:
        retval, rvec, tvec, inliers = cv2.solvePnPRansac(pose_3d, match_points, K, dist_coeffs, flags=cv2.USAC_MAGSAC, reprojectionError=8.0, iterationsCount=1000, confidence=0.999) 
    except AttributeError:
        retval, rvec, tvec, inliers = cv2.solvePnPRansac(pose_3d, match_points, K, dist_coeffs, flags=cv2.SOLVEPNP_EPNP, reprojectionError=8.0, iterationsCount=1000, confidence=0.999)
    
    if inliers is None or len(inliers) < 5:
        return {'B': None, 'L': None, 'H': None}, None, 0, [], reproj_error, float('inf')

    inliers = inliers.flatten()
    obj_pts_in = pose_3d[inliers]
    img_pts_in = match_points[inliers]

    # GA-PnP pose optimization
    def cost_function(params, obj_points, img_points, K, dist, target_pitch):
        r_vec = params[:3]
        t_vec = params[3:]
        
        # reprojection error
        proj_points, _ = cv2.projectPoints(obj_points, r_vec, t_vec, K, dist)
        residuals = (img_points - proj_points.squeeze()).ravel()
        
        # roll and pitch constraints
        R_mat, _ = cv2.Rodrigues(r_vec)
        R_wc = R_mat.T 
        
        ## roll constraint
        cam_right_z = R_wc[2, 0] 
        roll_residual = np.array([cam_right_z * 1000.0]) 
        extra_residuals = [roll_residual]

        ## pitch constraint
        if target_pitch is not None:
            cam_fwd = R_wc[:, 2] 
            up = cam_fwd[2]
            horiz = np.sqrt(cam_fwd[0]**2 + cam_fwd[1]**2)
            curr_pitch = np.degrees(np.arctan2(up, horiz))
            pitch_diff = curr_pitch - target_pitch
            pitch_residual = np.array([pitch_diff * 15.0]) 
            extra_residuals.append(pitch_residual)
            
        return np.concatenate([residuals] + extra_residuals)

    x0 = np.hstack((rvec.flatten(), tvec.flatten()))
    
    try:
        # optimize the pose
        res = least_squares(cost_function, x0, verbose=0, x_scale='jac', ftol=1e-4, method='trf', 
                            args=(obj_pts_in, img_pts_in, K, dist_coeffs, prior_pitch))
        
        # compute the covariance of the pose estimation (uncertainty estimation)
        try:
            J = res.jac 
            H = J.T @ J
            n_params = 6 
            n_residuals = len(res.fun)
            if n_residuals > n_params:
                mse = (2.0 * res.cost) / (n_residuals - n_params)
            else:
                mse = 1.0
            
            H_inv = np.linalg.inv(H)
            cov_matrix = H_inv * mse
            pos_variance = np.trace(cov_matrix[3:6, 3:6])
            pose_uncertainty = np.sqrt(pos_variance)
        except Exception as e:
            pose_uncertainty = float('inf')

        x_opt = res.x
        rvec, tvec = x_opt[:3].reshape(3, 1), x_opt[3:].reshape(3, 1)
        
        # cal the final reprojection error after optimization
        final_residuals = cost_function(x_opt, obj_pts_in, img_pts_in, K, dist_coeffs, prior_pitch)
        num_pixels = len(img_pts_in) * 2
        pixel_residuals = final_residuals[:num_pixels].reshape(-1, 2)
        reproj_error = np.mean(np.linalg.norm(pixel_residuals, axis=1))
        
    except Exception:
        pass

    # result conversion
    R1 = rotvector2rot(rvec)
    X0 = -R1.T @ tvec
    lon, lat = utm2deg(region, config, X0[0], X0[1])
    BLH = {'B': float(lat), 'L': float(lon), 'H': float(X0[2])}
    
    cam_angle = angles_from_pnp_rvec_heading_pitch(rvec) 
    
    return BLH, cam_angle, len(inliers), list(inliers), reproj_error, pose_uncertainty


def dumpRotateImage(img, degree):
    """rotate the image by degree"""
    height, width = img.shape[:2]
    radians = np.radians(degree)
    heightNew = int(abs(height * np.cos(radians)) + abs(width * np.sin(radians)))
    widthNew = int(abs(width * np.cos(radians)) + abs(height * np.sin(radians)))
    
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    if len(img.shape) == 3:
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        img_pil = Image.fromarray(img)

    img_rotated_pil = img_pil.rotate(degree, expand=True, resample=Image.BICUBIC)
    imgRotation = np.asarray(img_rotated_pil)
    
    if imgRotation.shape[0] != heightNew or imgRotation.shape[1] != widthNew:
       heightNew, widthNew = imgRotation.shape[:2]

    if len(imgRotation.shape) == 3:
        imgRotation = cv2.cvtColor(imgRotation, cv2.COLOR_RGB2BGR)

    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1.0)
    matRotation[0, 2] += (widthNew - width) / 2
    matRotation[1, 2] += (heightNew - height) / 2

    return imgRotation, matRotation

def rotvector2rot(rotvector):
    Rm = cv2.Rodrigues(rotvector)[0]
    return Rm

def utm2deg(region, config, x,y):
    utm_system = config[f'{region}_UTM_SYSTEM']
    transformer = Transformer.from_crs(f"epsg:326{utm_system[:2]}", "epsg:4326")
    lat, lon = transformer.transform(x, y)
    return lon, lat

def deg2utm(region, config, lon,lat):
    utm_system = config[f'{region}_UTM_SYSTEM']
    transformer = Transformer.from_crs("epsg:4326", f"epsg:326{utm_system[:2]}")
    x, y = transformer.transform(lat, lon)
    return x, y

def rot_to_euler(dcm, lim=None):
    r11 = -dcm[1][0]
    r12 = dcm[1][1]
    r21 = dcm[1][2]
    r31 = -dcm[0][2]
    r32 = dcm[2][2]
    r11a = dcm[0][1]
    r12a =  dcm[0][0]
    r1 = np.arctan2(r11, r12)
    r21 = np.clip(r21, -1, 1)  
    r2 = np.arcsin(r21)
    r3 = np.arctan2(r31, r32)
    if lim == 'zeror3':
        for i in np.where(np.abs(r21) == 1.0)[0]:
            r1[i] = np.arctan2(r11a[i], r12a[i])
            r3[i] = 0
    camAngle = np.array([-r1 - np.pi, -r2, r3 + np.pi]) * 180 / np.pi;
    return camAngle

def resolution_size(data, opt):
    """Estimate ground resolution GSD (m/pixel) and square crop size from camera parameters"""
    img_width = data['width']
    img_height = data['height']
    cam_size = data['cam_size']
    focal_len = data['focal_len']

    if opt.pose_priori == 'unknown':
        pitch = -90
    else:
        pitch = data['pitch']
    resolution = 2 * data['rel_height'] / np.sin(-np.pi * pitch / 180) * cam_size/2/focal_len \
                 / sqrt(img_width ** 2 + img_height ** 2)

    min_size = min(img_width, img_height) 
    size = np.array([min_size, min_size])
    return resolution, size


def crop_center(image_path, width, height):
    img = cv2.imread(image_path)
    original_height, original_width = img.shape[:2]
    center_x, center_y = original_width // 2, original_height // 2
    left = int(center_x - width // 2)
    top = int(center_y - height // 2)
    right = int(center_x + width // 2)
    bottom = int(center_y + height // 2)

    left = max(0, left)
    top = max(0, top)
    right = min(original_width, right)
    bottom = min(original_height, bottom)

    cropped_img = img[top:bottom, left:right]
    return Image.fromarray(cropped_img)

def find_values(config, search_string):
    results = 'None'
    for key, value in config.items():
        if search_string in key:
            results = value
    return results


def load_config_parameters_new(config, opt, region):
    Ref_type = opt.Ref_type
    t0 = time.time()
    # ref_image = cv2.imread(config[f'{region}_{Ref_type}_REF_PATH'])
    ref_image = cv2.cvtColor(tifffile.imread(config[f'{region}_{Ref_type}_REF_PATH']), cv2.COLOR_RGB2BGR)
    t1 = time.time()
    print(f'[timing] cv2.imread REF  : {(t1-t0)//60:.0f}分{(t1-t0)%60:.1f}秒')
    t0 = time.time()
    # dsm_image = cv2.imread(config[f'{region}_{Ref_type}_DSM_PATH'], cv2.IMREAD_UNCHANGED).astype(np.float32)
    dsm_image = tifffile.imread(config[f'{region}_{Ref_type}_DSM_PATH']).astype(np.float32)
    t1 = time.time()
    print(f'[timing] tifffile DSM    : {(t1-t0)//60:.0f}分{(t1-t0)%60:.1f}秒')
    ref_resolution = config[f'{region}_{Ref_type}_REF_resolution']
    save_path = opt.save_dir
    print('Result saved in : {}'.format(save_path))
    return ref_image, dsm_image, save_path, ref_resolution

def query_data_from_file(file_name, **query):
    data = read_data_from_file(file_name)
    results = []
    for entry in data:
        if all(entry.get(key) == value for key, value in query.items()):
            results.append(entry)
    return results

def read_data_from_file(file_name):
    if not os.path.exists(file_name):
        print("File does not exist!")
        return []
    with open(file_name, 'r') as f:
        data = json.load(f)
    return data

def matching_init(method_dict):
    method_name = method_dict['matching_method']
    
    if method_name == 'Roma':
        method_dict['matching_model'] = Roma_Init()
        
    elif method_name == 'RoMaV2':
        method_dict['matching_model'] = RoMaV2_Init()
        
    elif 'MINIMA' in method_name:
        # 例如: method_name 是 'MINIMA_Roma'
        # 传入 method_name 让 MINIMA_Init 内部去解析是哪种变体
        method_dict['matching_model'] = MINIMA_Init(method_name)
    else:
        print(f'Method name {method_name} is wrong or not supported!!!')
        
    return method_dict

def retrieval_init(method_dict, config, DINOv2_shared = None):
    if method_dict['retrieval_method'] not in config['RETRIEVAL_METHODS']:
        print('** The input model name is invalid. Please ensure you enter the correct model name. **')
        exit(1)
    else:
        method_dict['retrieval_model'], method_dict['img_transform'] = get_Model(method_dict['retrieval_method'], DINOv2_shared)
        method_dict['retrieval_model'].to(config['DEVICE'])
        method_dict['retrieval_cover'] = config['RETRIEVAL_COVER']
        method_dict['retrieval_topn'] = config['RETRIEVAL_TOPN']
        method_dict['retrieval_img_name'] = config['RETRIEVAL_IMG']
        method_dict['retrieval_feat_norm'] = config['RETRIEVAL_FEATURE_NORM']
    return method_dict

def retrieval_all(ref_image, UAV_path, uav_data, ref_resolution, matRotation, save_path, opt, region, config, method_dict):

    if 'CAMP' in method_dict['retrieval_method']:
        retrieval_size = 384  # CAMP must use the size of 384
    elif 'Roma' in method_dict['retrieval_method']:
        retrieval_size = 336  # the DINOv2 from the RoMa and MINIMA_RoMa use the size of 336
    else:
        retrieval_size = 336  # default size for other methods

    Ref_type = opt.Ref_type
    initialX = config[f'{region}_{Ref_type}_REF_initialX']
    initialY = config[f'{region}_{Ref_type}_REF_initialY']

    # the center of the uav query view in the reference image 
    center_x, center_y = view_center(region, config, uav_data, initialX, initialY, ref_resolution, matRotation)

    # calculate UAV query GSD and relative scale
    cover = method_dict['retrieval_cover']
    drone_resolution, drone_size = resolution_size(uav_data, opt) 
    finescale = drone_resolution / ref_resolution
    
    # make the retrieval block size adaptive to the drone's GSD, while use a opt.crop_gain to further adjust the block size
    view_size = drone_size * drone_resolution * opt.crop_gain  
    block_size = [math.ceil(view_size[0] / ref_resolution) + (math.ceil(view_size[0] / ref_resolution) % 2 != 0),
                  math.ceil(view_size[1] / ref_resolution) + (math.ceil(view_size[1] / ref_resolution) % 2 != 0)]
                  
    h_ref, w_ref = ref_image.shape[:2]
    if block_size[0] >= h_ref: block_size[0] = h_ref - 2
    if block_size[1] >= w_ref: block_size[1] = w_ref - 2
    
    block_size[0] = max(10, block_size[0])
    block_size[1] = max(10, block_size[1])

    step_size = [int(block_size[0] * (100 - cover) / 100), int(block_size[1] * (100 - cover)/100)]
    step_size[0] = max(1, step_size[0])
    step_size[1] = max(1, step_size[1])

    # formulate the candidate mid points for retrieval
    MID = compute_block_mid_wo_black(ref_image, block_size, step_size)
    if len(MID) == 0:
        MID = np.array([[h_ref // 2, w_ref // 2]])

    mids = MID.reshape(-1, 2)

    UAV_image = crop_center(UAV_path, drone_size[0], drone_size[1])

    retrieval_model = method_dict['retrieval_model']
    img_transform = method_dict['img_transform']
    retrieval_t0 = time.time()
    UAV_image = UAV_image.resize((retrieval_size, retrieval_size))
    UAV_img = img_transform(UAV_image)
    
    # gf: Gallery Features, qf: Query Feature, shifts: List of (sx, sy)
    gf, qf, shifts = extract_features(retrieval_size, config, method_dict, MID, block_size, ref_image, UAV_img, img_transform, retrieval_model)

    retrieval_t1 = time.time()

    score = gf @ qf.unsqueeze(-1)
    score = score.squeeze().cpu().numpy()
    
    if score.ndim == 0:
        order = np.array([0])
        sorted_score = np.array([float(score)])
    else:
        order = np.argsort(score)  # from small to large
        order = order[::-1] # from large to small
        sorted_score = score[order]

    retrieval_time_cost = (retrieval_t1 - retrieval_t0)/(len(MID)+1)

    best_start_x = []
    best_start_y = []
    best_H_list = [] 
    best_W_list = [] 
    PDE_list = []
    
    for i in range(len(order)):
        idx = order[i]
        mid_x, mid_y = mids[idx, :]
        
        current_H = block_size[0]
        current_W = block_size[1]

        if shifts is not None and idx < len(shifts):
            shift_data = shifts[idx]
            
            if len(shift_data) == 3:
                s_col, s_row, s_scale = shift_data
            elif len(shift_data) == 2:
                s_col, s_row = shift_data
                s_scale = 1.0 
            else:
                s_col, s_row, s_scale = 0.0, 0.0, 1.0

            current_H = int(block_size[0] * s_scale)
            current_W = int(block_size[1] * s_scale)

            offset_row = s_row * block_size[0]
            offset_col = s_col * block_size[1]
            
            mid_x += offset_row
            mid_y += offset_col
            
            mid_x = np.clip(mid_x, current_H/2, h_ref - current_H/2)
            mid_y = np.clip(mid_y, current_W/2, w_ref - current_W/2)
        
        start_x = max(0, int(mid_x - current_H / 2))
        start_y = max(0, int(mid_y - current_W / 2))
        
        best_start_x.append(start_x)
        best_start_y.append(start_y)
        best_H_list.append(current_H)
        best_W_list.append(current_W)

        d_i = ((center_x - mid_y) ** 2 + (center_y - mid_x) ** 2) ** 0.5
        p_i = d_i / block_size[0]
        PDE_list.append(p_i)

    show_n = config['SHOW_RETRIEVAL_IMG_NUM']
    
    # shown retrieval results
    if config['SHOW_RETRIEVAL_RESULT']:
        retrieval_img_name = method_dict['retrieval_img_name']
        fig = plt.figure(figsize=(16, 4))
        ax0 = plt.subplot(1, show_n + 1, 1)
        ax0.axis('off')
        UAV_image_1 = np.array(UAV_image)
        plt.imshow(UAV_image_1[..., ::-1])
        plt.title('UAV')
        for i in range(min(len(mids), show_n)):
            ax = plt.subplot(1, show_n + 1, i + 2)
            ax.axis('off')
            
            x_s = best_start_x[i]
            y_s = best_start_y[i]
            x_e = int(x_s + block_size[0])
            y_e = int(y_s + block_size[1])

            x_e = min(h_ref, x_e)
            y_e = min(w_ref, y_e)
            
            img = ref_image[x_s:x_e, y_s:y_e]
            if img.size == 0: continue 

            if i == 0:
                middle_position = (ax0.get_position().xmax + ax.get_position().xmin) / 2
                fig.add_artist(
                    plt.Line2D([middle_position, middle_position], [0.25, 0.75], color='#CD2626',
                               linestyle='dashed'))
            plt.imshow(img[..., ::-1])
            ax.set_title(f'{i + 1} PDE:{PDE_list[i]:.3f}', color='#1C86EE', fontsize=20)

        plt.ioff()
        save_dir = os.path.dirname(save_path + retrieval_img_name)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path + retrieval_img_name)
        plt.close(fig)

    # shown retrieval shift trajectories
    if config['SHOW_RETRIEVAL_RESULT']:
        vis_count = config['SHOW_RETRIEVAL_IMG_NUM'] 
        
        gt_rc = (center_y, center_x) 

        for rank in range(min(len(order), vis_count)):
            idx = order[rank] 
            
            orig_r, orig_c = mids[idx, :] 
            
            curr_h = best_H_list[rank]
            curr_w = best_W_list[rank]
            
            new_r = best_start_x[rank] + curr_h / 2.0
            new_c = best_start_y[rank] + curr_w / 2.0
            
            retrieval_img_name = method_dict['retrieval_img_name'] 
            base_dir = os.path.dirname(save_path + retrieval_img_name)
            uav_name = os.path.basename(UAV_path).split('.')[0]
            
            viz_name = f"Traj_Rank{rank+1}_{uav_name}.png"
            traj_save_path = os.path.join(base_dir, viz_name)
            
            draw_shift_trajectory(
                ref_image=ref_image,
                gt_rc=gt_rc,
                old_center_rc=(orig_r, orig_c),  
                new_center_rc=(new_r, new_c),    
                old_size=block_size,             
                new_size=(curr_h, curr_w),      
                save_path=traj_save_path
            )


        print(f"Top-{vis_count} Trajectory visualizations saved to {base_dir}")
    return order, best_start_x, best_start_y, PDE_list, best_W_list, best_H_list, finescale, retrieval_time_cost, sorted_score

def compute_block_mid_wo_black(image, block_size, step_size):
    """compute retrieval block mid points while avoiding black borders in the reference image"""
    # use thumbnails to increase computing speed
    scale = 5
    thresold = 0.9 # valid pixel threshold to filter out black borders
    h, w = image.shape[:2]
    
    target_h = max(1, int(h / scale))
    target_w = max(1, int(w / scale))
    small_img = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    def get_clean_starts(length, blk_len, step):
        starts = list(range(0, length - blk_len + 1, step))
        if len(starts) == 0 or starts[-1] < length - blk_len:
            starts.append(length - blk_len)
        return sorted(list(set(starts)))

    x_starts = get_clean_starts(h, block_size[0], step_size[0])
    y_starts = get_clean_starts(w, block_size[1], step_size[1])

    mids = []
    
    for start_x in x_starts:
        for start_y in y_starts:
            end_x = start_x + block_size[0]
            end_y = start_y + block_size[1]
            
            s_sx, s_ex = int(start_x / scale), int(end_x / scale)
            s_sy, s_ey = int(start_y / scale), int(end_y / scale)
            
            if s_ex <= s_sx: s_ex = s_sx + 1
            if s_ey <= s_sy: s_ey = s_sy + 1
            
            block_small = small_img[s_sx:s_ex, s_sy:s_ey]
            if block_small.size == 0: continue
            
            if len(block_small.shape) == 3:
                valid_mask = np.any(block_small > 0, axis=-1)
            else:
                valid_mask = block_small > 0
                
            valid_ratio = np.sum(valid_mask) / valid_mask.size
            
            if valid_ratio >= thresold:
                mid_x = (start_x + end_x) / 2.0
                mid_y = (start_y + end_y) / 2.0
                mids.append([mid_x, mid_y])

    if len(mids) == 0:
        return np.array([[h/2.0, w/2.0]])
        
    return np.array(mids)

def visualize_final_localization(ref_img, uav_img, true_pos, pred_pos, config, region, save_path, opt):
    """
        visualize the final localization result
    """

    try:
        vis_img = ref_img.copy()
        h, w = vis_img.shape[:2]
        
        base_dir = os.path.dirname(save_path)

        # save init uav query image
        if uav_img is not None:
            cv2.imwrite(os.path.join(base_dir, 'final_uav_raw.jpg'), uav_img)
        
        # save satellite reference image
        cv2.imwrite(os.path.join(base_dir, 'final_sat_raw.jpg'), ref_img)

        Ref_type = opt.Ref_type
        initialX = config[f'{region}_{Ref_type}_REF_initialX']
        initialY = config[f'{region}_{Ref_type}_REF_initialY']
        ref_res = config[f'{region}_{Ref_type}_REF_resolution']

        # [SCI Colors]
        COLOR_GT = (0, 255, 127)    
        COLOR_PRED = (60, 20, 220)  
        COLOR_LINE = (255, 255, 0)  
        COLOR_FOV = (0, 255, 255)   
        
        MARKER_SIZE = 12
        ARROW_LEN = 45

        # cal GT coordinate
        t_utm_x, t_utm_y = deg2utm(region, config, true_pos['lon'], true_pos['lat'])
        t_col = int((t_utm_x - initialX) / ref_res)
        t_row = int((initialY - t_utm_y) / ref_res)
        
        # cal Pred coordinate
        has_pred = (pred_pos.get('lon') is not None)
        if has_pred:
            p_utm_x, p_utm_y = deg2utm(region, config, pred_pos['lon'], pred_pos['lat'])
            p_col = int((p_utm_x - initialX) / ref_res)
            p_row = int((initialY - p_utm_y) / ref_res)
        else:
            p_col, p_row = -1, -1

        # draw FoV and crop satellite image
        if 0 <= t_col < w and 0 <= t_row < h:
            uav_res, uav_size_px = resolution_size(true_pos, opt) 
            
            scale_factor = uav_res / ref_res
            
            box_w_sat = int(uav_size_px[1] * scale_factor)
            box_h_sat = int(uav_size_px[0] * scale_factor)
            
            x1 = int(t_col - box_w_sat / 2)
            y1 = int(t_row - box_h_sat / 2)
            x2 = int(t_col + box_w_sat / 2)
            y2 = int(t_row + box_h_sat / 2)
            
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), COLOR_FOV, 2, cv2.LINE_AA)
            cv2.putText(vis_img, "FOV", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_FOV, 1)

            c_x1, c_y1 = max(0, x1), max(0, y1)
            c_x2, c_y2 = min(w, x2), min(h, y2)
            
            if c_x2 > c_x1 and c_y2 > c_y1:
                sat_fov_crop = ref_img[c_y1:c_y2, c_x1:c_x2]
                cv2.imwrite(os.path.join(base_dir, 'final_sat_fov_crop.jpg'), sat_fov_crop)
        # ============================================================

        # draw error line
        if has_pred and 0 <= t_col < w and 0 <= t_row < h and 0 <= p_col < w and 0 <= p_row < h:
            cv2.line(vis_img, (t_col, t_row), (p_col, p_row), COLOR_LINE, 2, cv2.LINE_AA)

        # draw GT marker
        if 0 <= t_col < w and 0 <= t_row < h:
            cv2.circle(vis_img, (t_col, t_row), MARKER_SIZE, COLOR_GT, 2, cv2.LINE_AA)
            cv2.drawMarker(vis_img, (t_col, t_row), COLOR_GT, cv2.MARKER_CROSS, 15, 2, cv2.LINE_AA)
            
            if 'yaw' in true_pos and true_pos['yaw'] is not None:
                yaw_rad = np.deg2rad(true_pos['yaw'])
                dx, dy = np.sin(yaw_rad), -np.cos(yaw_rad)
                end_x = int(t_col + ARROW_LEN * dx)
                end_y = int(t_row + ARROW_LEN * dy)
                cv2.arrowedLine(vis_img, (t_col, t_row), (end_x, end_y), COLOR_GT, 2, cv2.LINE_AA, tipLength=0.25)
            
            cv2.putText(vis_img, "GT", (t_col + 15, t_row - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_GT, 2)

        # draw Pred marker
        if has_pred and 0 <= p_col < w and 0 <= p_row < h:
            cv2.circle(vis_img, (p_col, p_row), 6, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(vis_img, (p_col, p_row), 4, COLOR_PRED, -1, cv2.LINE_AA)
            
            if 'yaw' in pred_pos and pred_pos['yaw'] is not None:
                yaw_rad = np.deg2rad(pred_pos['yaw'])
                dx, dy = np.sin(yaw_rad), -np.cos(yaw_rad)
                end_x = int(p_col + ARROW_LEN * dx)
                end_y = int(p_row + ARROW_LEN * dy)
                cv2.arrowedLine(vis_img, (p_col, p_row), (end_x, end_y), COLOR_PRED, 2, cv2.LINE_AA, tipLength=0.25)
            
            cv2.putText(vis_img, "Pred", (p_col + 15, p_row + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_PRED, 2)
        
        # save final visualization image
        cv2.imwrite(save_path, vis_img)
        
    except Exception as e:
        print(f"Visualization Warning: {e}")
        import traceback
        traceback.print_exc()


def draw_shift_trajectory(ref_image, gt_rc, old_center_rc, new_center_rc, old_size, new_size, save_path):
    """
    draw the shift trajectory and elastic box changes
    """
    vis_img = ref_image.copy()
    H, W = vis_img.shape[:2]
    
    def draw_box(img, center_rc, box_size, color, thickness=2, style='solid'):
        r, c = center_rc
        h_curr, w_curr = box_size
        
        pt1 = (int(c - w_curr/2), int(r - h_curr/2)) # left up (Col, Row)
        pt2 = (int(c + w_curr/2), int(r + h_curr/2)) # right down (Col, Row)
        
        pt1 = (max(0, pt1[0]), max(0, pt1[1]))
        pt2 = (min(W, pt2[0]), min(H, pt2[1]))
        
        cv2.rectangle(img, pt1, pt2, color, thickness)
        return (int(c), int(r))
    
    # draw init box
    pt_old = draw_box(vis_img, old_center_rc, old_size, (255, 200, 0), thickness=2)
    
    # draw shift box
    pt_new = draw_box(vis_img, new_center_rc, new_size, (0, 0, 255), thickness=4)
    
    # draw shift arrow
    if pt_old != pt_new:
        cv2.arrowedLine(vis_img, pt_old, pt_new, (0, 255, 255), thickness=3, line_type=cv2.LINE_AA, tipLength=0.2)
    
    # draw GT marker
    gt_r, gt_c = int(gt_rc[0]), int(gt_rc[1])
    if 0 <= gt_r < H and 0 <= gt_c < W:
        cv2.drawMarker(vis_img, (gt_c, gt_r), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=30, thickness=3)
        cv2.putText(vis_img, "GT", (gt_c + 10, gt_r - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # draw shift info
    scale_ratio = new_size[0] / old_size[0]
    info_text = f"Shifted | Scale: {scale_ratio:.2f}x"
    
    # draw legend
    cv2.putText(vis_img, "Blue: Orig / Red: Elastic Shift", (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis_img, info_text, (20, 65), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # save image
    cv2.imwrite(save_path, vis_img)
