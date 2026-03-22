import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import argparse
import pandas as pd
import yaml
import glob
from utils import (find_values, get_jpg_files, load_config_parameters_new, 
                   computeCameraMatrix, crop_geo_images_with_offset, dumpRotateImage, 
                   retrieval_init, matching_init, retrieval_all, Match2Pos_all, 
                   pos2error, visualize_final_localization, save_data, 
                   query_data_from_file, img_name, read_data_from_file,
                   normalize_angle) 
import warnings
from tqdm import tqdm
import cv2
import numpy as np
import time
import json
from datetime import datetime
warnings.filterwarnings("ignore")
import torch
import random
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def seed_everything(seed=42):
    """
    Fix all possible random sources to ensure the reproducibility of the experiment
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    import cv2
    cv2.setRNGSeed(seed)

seed_everything(42)

# Define the time suffix for the output folder
time_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")

# Define the list of query images you want to run (called as whitelist easy to visualize)
target_imgs = [
    # "DJI_202512212147_018_train_changsha_city_300_ortho_night_DJI_20251221222009_0483_V.JPG"
    # "DJI_202512171221_003_train_changsha_village_300_ortho_day_DJI_20251217123807_0063_V.JPG"
    # "DJI_202512211108_015_test_changsha_city1_300_ortho_day_DJI_20251221111419_0016_T.JPG"
]

def get_parse():
    parser = argparse.ArgumentParser(description='UAV-Visual-Localization')
    parser.add_argument('--yaml', default='config.yaml', type=str, help='Retrieval and Matching configuration yaml file')
    parser.add_argument('--save_dir', default=r'./Result/Experiment', type=str, help='The root directory to save the results')
    parser.add_argument('--device', default='cuda', type=str, help='Inference device')
    parser.add_argument('--pose_priori', default='yp', type=str,
                        help='prior about the pose, yp: yaw and pitch, p: pitch, unknown: no pitch and yaw')
    parser.add_argument('--Ref_type', default='HIGH', type=str, help='HIGH(LOW): it is the resolution of reference map while we only offer the High type in the benchmark')
    parser.add_argument('--resize_ratio', default=0.3, type=float, help='To save inference time and memory in image matching stage') 
    parser.add_argument('--crop_size_m', default=600, type=float, help='crop the size of the satellite reference map, in meter')
    parser.add_argument('--crop_gain', default=1.5, type=float, help='GSD scaling factor, adjust the retrieve crop size') 
    
    parser.add_argument('--noise_yaw', default=0, type=float, help='Uniform noise range [-val, val] for Yaw (deg)')
    parser.add_argument('--noise_pitch', default=0, type=float, help='Uniform noise range [-val, val] for Pitch (deg)')
                                                         
    opt = parser.parse_args()
    
    print(opt)
    return opt


if __name__ == '__main__':
    # load method-specific config
    opt = get_parse()
    opt.save_dir = f"{opt.save_dir}_{time_suffix}"

    with open(opt.yaml, 'r') as f:
        config = yaml.safe_load(f)
    
    All_Region = config['REGIONS']
    All_Retrieval = config['RETRIEVAL_METHODS']
    All_Matching = config['MATCHING_METHODS']
    split = config['SPLIT'] # train, test, valid. The test set is used for the final evaluation.
    modality = config['MODALITY'] # Thermal, Visible. In the benchmark, we only provide thermal images as query.
    
    # store the results
    all_rows = []  
    excel_columns = [
        'image_name', 'pred_err_horz(m)', 
        'err_lat(m)', 'err_lon(m)', 'err_alt(m)', 
        'err_roll(deg)', 'err_pitch(deg)', 'err_yaw(deg)',
        'pred_lat', 'pred_lon', 'pred_alt', 
        'pred_pitch', 'pred_yaw', 'time_cost(s)'
    ]
    
    # loop through each region and perform localization (In the benchmark we only offer the changsha region)
    for region in All_Region:
        
        # load region-specific config and query image metadata
        yaml_file = f'./Regions_params/{region}.yaml'
        with open(yaml_file, 'r') as f:
            region_config = yaml.safe_load(f)
        region_config.update(config)

        region_path =  f"./Data/Thermal-UAV/{split}/{region}"
        json_file = f'./Data/metadata/{split}_{modality}.json'
        
        metadata_cache = {}
        _meta_list = read_data_from_file(json_file)
        for _item in _meta_list:
            if 'name' in _item:
                metadata_cache[_item['name']] = _item

        UAV_img_list0 = []
        for place in os.listdir(region_path):
            img_modality_path = f"{region_path}/{place}/{modality}" 
            UAV_img_list0 += [f.replace('\\', '/') for f in glob.glob(os.path.join(img_modality_path, '*.JPG'))]
            
        # use the 'TEST_INTERVAL' to adjust the number of test samples
        UAV_img_list = UAV_img_list0[0::region_config['TEST_INTERVAL']]
        
        # if use whitelist, filter the UAV_img_list by the target_imgs
        if len(target_imgs) > 0:
            UAV_img_list = [p for p in UAV_img_list if os.path.basename(p) in target_imgs]
            
            print(f"\n[INFO] whitelist is enabled, only processing {len(UAV_img_list)} images:")
            
            for p in UAV_img_list:
                print(f"  -> {os.path.basename(p)}")
            
            if len(UAV_img_list) == 0:
                print("[WARNING] No matching images found. Please check the file names for accuracy!")

        method_dict = {} # a dictionary to store the method-specific parameters
        
        # loop through each retrieval method
        for retrieval_index in range(len(All_Retrieval)):
            
            # skip these two retrieval methods since they share the DINOv2 backbone with the matching stage and require special handling
            if All_Retrieval[retrieval_index] != 'MINIMA_Roma_DINOv2' and All_Retrieval[retrieval_index] != 'Roma_DINOv2':
                # Retrieval method
                method_dict['retrieval_method'] = All_Retrieval[retrieval_index]
                method_dict = retrieval_init(method_dict, region_config) 
                
            # loop through each matching method
            for match_index in range(len(All_Matching)):
                # Matching method
                method_dict['matching_method'] = All_Matching[match_index]
                method_dict = matching_init(method_dict)

                # special handling for the MINIMA_Roma_DINOv2 retrieval method and the Roma_DINOv2 retrieval method since they share the DINOv2 backbone with the matching stage
                # the duplicate use of DINOv2 backbone will lower the memory usage
                if All_Retrieval[retrieval_index] == 'MINIMA_Roma_DINOv2':
                    method_dict['retrieval_method'] = All_Retrieval[retrieval_index]
                    method_dict = retrieval_init(method_dict, region_config, DINOv2_shared=method_dict['matching_model'].model.encoder.dinov2_vitl14[0]) 
                elif All_Retrieval[retrieval_index] == 'Roma_DINOv2':
                    method_dict['retrieval_method'] = All_Retrieval[retrieval_index]
                    method_dict = retrieval_init(method_dict, region_config, DINOv2_shared=method_dict['matching_model'].encoder.dinov2_vitl14[0])

                # read the 2D reference map as well as the corresponding DSM data
                ref_map_full, dsm_map_full, save_path0, ref_resolution = load_config_parameters_new(region_config, opt, region)

                # deal withthe UAV query images one by one
                for index, uav_path in enumerate(tqdm(UAV_img_list, desc=f"{region}", unit="image")):
                    place = os.path.basename(os.path.dirname(os.path.dirname(uav_path)))
                    print('Region: {} Place: {} Pic: {} Ratio: {:.1f}%'.format(region, place, os.path.basename(uav_path),
                                                                    index / len(UAV_img_list) * 100))
                    
                    VG_pkl_path = '{}/{}/pkl_{}/VG_data_{}.pkl'. format(opt.save_dir, region, place, img_name(uav_path))
                    if os.path.exists(VG_pkl_path):
                        continue

                    save_path = '{}/{}/{}'.format(save_path0, place, img_name(uav_path))
                    os.makedirs(save_path, exist_ok=True) # 确保文件夹已创建
                    
                    if uav_path in metadata_cache:
                        truePos = metadata_cache[uav_path]
                    else:
                        raise IndexError(f"Query name {uav_path} not found in metadata.")

                    noisyPos = truePos.copy()
                    if opt.noise_yaw > 0 or opt.noise_pitch > 0:
                        # uniform noise for yaw and pitch
                        yaw_noise = np.random.uniform(-opt.noise_yaw, opt.noise_yaw)
                        pitch_noise = np.random.uniform(-opt.noise_pitch, opt.noise_pitch)
                        
                        noisyPos['yaw'] = normalize_angle(truePos['yaw'] + yaw_noise)
                        noisyPos['pitch'] = np.clip(truePos['pitch'] + pitch_noise, -90.0, 90.0) # use [-90, 90] range for pitch
                    
                    K = computeCameraMatrix(truePos)
                    uav_image = cv2.imread(uav_path)

                    # crop the reference map and DSM
                    ref_map0, dsm_map0, temp_config = crop_geo_images_with_offset(
                        region, region_config, opt, noisyPos, ref_map_full, dsm_map_full
                    )

                    # use the noisy prior to rotate the reference map
                    if opt.pose_priori == 'yp':
                        ref_map, matRotation = dumpRotateImage(ref_map0, noisyPos['yaw'])
                    else:
                        ref_map = ref_map0
                        matRotation = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

                    VG_time0 = time.time()

                    prior_pitch_value = noisyPos['pitch']
                    # Step 1: Retrieval
                    IR_order, refLocX, refLocY, PDE_list, cut_W_list, cut_H_list, fineScale, retrieval_time, sorted_score = retrieval_all(
                        ref_map, uav_path, noisyPos, ref_resolution, matRotation, save_path, opt, region, temp_config, method_dict
                    )

                    # Step 2: Matching
                    BLH_list, inliners_list, match_time, pnp_time, reproj_error_list, retrieval_score_subset, angle_list, uncertainty_list = Match2Pos_all(
                        opt, region, temp_config, uav_image, fineScale, K, ref_map, dsm_map0, refLocY, refLocX, 
                        cut_H_list, cut_W_list, 
                        save_path, method_dict, matRotation, 
                        retrieval_scores=sorted_score,
                        prior_pitch=prior_pitch_value
                    )
                    
                    # Step 3: error calculation and Position Selection
                    detailed_error_dict, pred_error_horz, _ = pos2error(
                        truePos, BLH_list, inliners_list, reproj_error_list, retrieval_score_subset, angle_list, uncertainty_list
                    )

                    VG_time_cost = time.time() - VG_time0
                    
                    # print the result and store in row to be saved in excel
                    print('pred_error_horz: {:.4f} m'.format(pred_error_horz))
                    if detailed_error_dict:
                        pred_yaw = detailed_error_dict.get('pred_yaw', float('nan'))
                        gt_yaw = truePos.get('yaw', float('nan'))
                        
                        print(f"  -> Yaw Comparison | Pred: {pred_yaw:6.2f}° | GT: {gt_yaw:6.2f}° | Err: {detailed_error_dict['err_yaw']:6.2f}°")
                        
                        print("------------------------------------------------------------------------------")

                        pred_pos = {
                            'lon': detailed_error_dict['pred_lon'],
                            'lat': detailed_error_dict['pred_lat'],
                            'yaw': detailed_error_dict.get('pred_yaw') 
                        }
                        
                        row = [
                            img_name(uav_path), pred_error_horz,
                            detailed_error_dict['err_lat'], detailed_error_dict['err_lon'], detailed_error_dict['err_alt'],
                            detailed_error_dict['err_roll'], detailed_error_dict['err_pitch'], detailed_error_dict['err_yaw'],
                            detailed_error_dict['pred_lat'], detailed_error_dict['pred_lon'], detailed_error_dict['pred_alt'],
                            detailed_error_dict['pred_pitch'], detailed_error_dict['pred_yaw'], VG_time_cost
                        ]
                        all_rows.append(row)
                    else:
                        print(f"  -> Localization Failed for {img_name(uav_path)}")
                        pred_pos = {'lon': None, 'lat': None, 'yaw': None}
                        row = [img_name(uav_path)] + [None]*12 + [VG_time_cost]
                        all_rows.append(row)
                        
                    # Visualization
                    vis_save_path = os.path.join(save_path, 'final_result_vis.jpg')
                    if config['SHOW_RETRIEVAL_RESULT']:
                        visualize_final_localization(ref_map0, uav_image, truePos, pred_pos, temp_config, region, vis_save_path, opt)
                        
                    # Save to pkl for detailed analysis and debugging
                    save_data(VG_pkl_path, 
                              opt=opt, 
                              region_config=temp_config, 
                              img_path=uav_path, 
                              truePos=truePos, 
                              noisyPos=noisyPos, 
                              refLocX=refLocX, 
                              refLocY=refLocY, 
                              IR_order=IR_order, 
                              PDE=PDE_list, 
                              inliners=inliners_list, 
                              BLH_list=BLH_list,
                              detailed_error_dict=detailed_error_dict, # 存详细误差
                              reproj_error_list=reproj_error_list,
                              angle_list=angle_list,
                              retrieval_time=retrieval_time, 
                              match_time=match_time, 
                              pnp_time=pnp_time, 
                              total_time=VG_time_cost)
                    
    # Save all results to one excel to easy to check
    if len(all_rows) > 0:
        df = pd.DataFrame(all_rows, columns=excel_columns)
        os.makedirs(opt.save_dir, exist_ok=True)
        excel_path = os.path.join(opt.save_dir, 'Detailed_6DOF_Results.xlsx')
        df.to_excel(excel_path, index=False)
        print(f'All results saved to: {excel_path}')
    else:
        print('No results to save')
