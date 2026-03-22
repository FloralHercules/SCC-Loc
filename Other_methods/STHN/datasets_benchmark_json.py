import os
import random
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import tifffile
import torch
import torch.utils.data as data
from PIL import Image
from pyproj import Transformer
import torchvision.transforms.functional as F
import yaml

from datasets_4cor_img import homo_dataset, seed_worker


def _normalize_path(raw_path: str, root_dir: Path) -> str:
    rel = raw_path.replace('\\', '/').strip()
    if rel.startswith('./'):
        rel = rel[2:]
    return str((root_dir / rel).resolve())


def _extract_region(image_path: str):
    path = image_path.replace('\\', '/')
    marker = '/Thermal-UAV/'
    if marker not in path:
        return None
    tail = path.split(marker, 1)[1]
    parts = tail.split('/')
    if len(parts) < 3:
        return None
    return parts[1]


def _utm_epsg_from_system(utm_system: str):
    zone = int(str(utm_system)[:-1])
    hemi = str(utm_system)[-1].upper()
    return 32600 + zone if hemi == 'N' else 32700 + zone


def _deg2utm(region: str, config: dict, lon: float, lat: float):
    epsg = _utm_epsg_from_system(config[f'{region}_UTM_SYSTEM'])
    transformer = Transformer.from_crs('EPSG:4326', f'EPSG:{epsg}', always_xy=True)
    x, y = transformer.transform(float(lon), float(lat))
    return float(x), float(y)


def _get_ground_target_lonlat(sample_query: dict):
    target_lon = sample_query.get('TargetLongitude', None)
    target_lat = sample_query.get('TargetLatitude', None)
    if target_lon is not None and target_lat is not None:
        return float(target_lon), float(target_lat)
    return float(sample_query['lon']), float(sample_query['lat'])


def _crop_geo_images_with_offset_same_as_baseline(region, config, ref_type, crop_size_m, offset_range_m, true_pos, ref_map_full):
    orig_initial_x = config[f'{region}_{ref_type}_REF_initialX']
    orig_initial_y = config[f'{region}_{ref_type}_REF_initialY']
    ref_res = config[f'{region}_{ref_type}_REF_resolution']
    orig_ref_coor = config[f'{region}_{ref_type}_REF_COORDINATE']

    uav_utm_x, uav_utm_y = _deg2utm(region, config, true_pos['lon'], true_pos['lat'])

    half_size = crop_size_m / 2.0
    offset_x = random.uniform(-offset_range_m, offset_range_m)
    offset_y = random.uniform(-offset_range_m, offset_range_m)
    center_crop_x = uav_utm_x + offset_x
    center_crop_y = uav_utm_y + offset_y

    tl_crop_utm_x = center_crop_x - half_size
    tl_crop_utm_y = center_crop_y + half_size

    ref_col_start = int((tl_crop_utm_x - orig_initial_x) / ref_res)
    ref_row_start = int((orig_initial_y - tl_crop_utm_y) / ref_res)
    ref_w_pixel = int(crop_size_m / ref_res)
    ref_h_pixel = int(crop_size_m / ref_res)
    ref_col_end = ref_col_start + ref_w_pixel
    ref_row_end = ref_row_start + ref_h_pixel

    h_ref, w_ref = ref_map_full.shape[:2]
    ref_col_start = max(0, ref_col_start)
    ref_row_start = max(0, ref_row_start)
    ref_col_end = min(w_ref, ref_col_end)
    ref_row_end = min(h_ref, ref_row_end)
    ref_crop = ref_map_full[ref_row_start:ref_row_end, ref_col_start:ref_col_end]

    new_config = config.copy()
    new_ref_initial_x = orig_initial_x + ref_col_start * ref_res
    new_ref_initial_y = orig_initial_y - ref_row_start * ref_res
    new_config[f'{region}_{ref_type}_REF_initialX'] = new_ref_initial_x
    new_config[f'{region}_{ref_type}_REF_initialY'] = new_ref_initial_y
    new_config[f'{region}_{ref_type}_REF_COORDINATE'] = [orig_ref_coor[0] - ref_col_start, orig_ref_coor[1] - ref_row_start]

    return ref_crop, None, new_config


def _sat_crop_to_tensor(ref_crop_bgr: np.ndarray, output_size: int):
    if ref_crop_bgr.ndim == 2:
        ref_crop = cv2.cvtColor(ref_crop_bgr, cv2.COLOR_GRAY2RGB)
    else:
        ref_crop = ref_crop_bgr
    img_t = F.to_tensor(np.asarray(ref_crop))
    img_t = F.resize(img_t, [output_size, output_size], antialias=True)
    return img_t


def _project_point_after_center_crop_resize(x, y, width, height, out_size):
    side = min(height, width)
    left = (width - side) / 2.0
    top = (height - side) / 2.0
    scale = float(out_size) / float(side)
    x_new = (x - left) * scale
    y_new = (y - top) * scale
    return x_new, y_new


def _center_square_crop_pil(img: Image.Image) -> Image.Image:
    width, height = img.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return img.crop((left, top, left + side, top + side))


def _extract_rotation_prior_deg(sample_query: dict, default_deg: float = 0.0) -> float:
    candidate_keys = [
        'rotation', 'rotation_deg', 'yaw', 'heading', 'course', 'bearing',
        'drone_yaw', 'uav_yaw', 'angle', 'TargetYaw', 'target_yaw'
    ]
    for key in candidate_keys:
        value = sample_query.get(key, None)
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            continue
    return float(default_deg)


def _rotate_query_with_inscribed_crop(query_img: Image.Image, degree: float, keep_size: bool = True) -> Image.Image:
    if abs(float(degree)) <= 1e-6:
        return query_img

    img_sq = _center_square_crop_pil(query_img)
    side = img_sq.size[0]
    rotated = img_sq.rotate(float(degree), expand=False, resample=Image.BICUBIC)

    crop_side = max(2, int(round(float(side) / np.sqrt(2.0))))
    left = (side - crop_side) // 2
    top = (side - crop_side) // 2
    cropped = rotated.crop((left, top, left + crop_side, top + crop_side))

    if keep_size and crop_side != side:
        cropped = cropped.resize((side, side), resample=Image.BICUBIC)
    return cropped


def _is_black_area_on_tensor(img_t: torch.Tensor, x: float, y: float, radius: int = 3,
                             zero_threshold: float = 0.03, zero_ratio_threshold: float = 0.98) -> bool:
    if img_t is None or img_t.ndim != 3:
        return False
    _, h, w = img_t.shape
    x_i = int(round(float(x)))
    y_i = int(round(float(y)))
    if x_i < 0 or x_i >= w or y_i < 0 or y_i >= h:
        return True

    x0 = max(0, x_i - int(radius))
    x1 = min(w, x_i + int(radius) + 1)
    y0 = max(0, y_i - int(radius))
    y1 = min(h, y_i + int(radius) + 1)
    patch = img_t[:, y0:y1, x0:x1]
    if patch.numel() == 0:
        return True

    gray = patch.mean(dim=0)
    black_ratio = float((gray <= float(zero_threshold)).float().mean().item())
    return black_ratio >= float(zero_ratio_threshold)


class BenchmarkJsonDataset(homo_dataset):
    def __init__(
        self,
        args,
        split='train',
    ):
        super().__init__(args, augment=(args.augment == 'img'))
        self.args = args
        self.split = split
        self.root_dir = Path(args.root_dir).resolve()
        self.ref_type = args.Ref_type
        self.sat_size_px = int(getattr(args, 'sat_size_px', max(2, int(round(float(args.crop_size_m))))))

        thermal_json = {
            'train': args.train_thermal_json,
            'val': args.val_thermal_json,
            'test': args.test_thermal_json,
            'extended': args.train_thermal_json,
        }[split]

        with open((self.root_dir / thermal_json).resolve(), 'r', encoding='utf-8') as f:
            thermal_items = json.load(f)

        with open((self.root_dir / args.global_config_yaml).resolve(), 'r', encoding='utf-8') as f:
            global_cfg = yaml.safe_load(f)
        self.global_cfg = global_cfg
        self.region_meta = {}
        self.region_runtime_cache = {}

        def ensure_region_meta_loaded(region_name: str):
            if region_name in self.region_meta:
                return

            region_yaml = (self.root_dir / args.region_yaml_dir / f'{region_name}.yaml').resolve()
            with open(region_yaml, 'r', encoding='utf-8') as rf:
                region_cfg = yaml.safe_load(rf)
            region_cfg.update(self.global_cfg)

            ref_path = (self.root_dir / region_cfg[f'{region_name}_{self.ref_type}_REF_PATH'].replace('\\', '/').lstrip('./')).resolve()
            self.region_meta[region_name] = {
                'config': region_cfg,
                'ref_path': str(ref_path),
            }

        self.samples = []
        for index, query in enumerate(thermal_items):
            query2 = dict(query)
            query2['abs_path'] = _normalize_path(query['name'], self.root_dir)
            region_name = _extract_region(query['name'])
            if region_name is None:
                raise ValueError(f"Invalid query path format (cannot parse region): {query['name']}")
            ensure_region_meta_loaded(region_name)
            self.samples.append(
                {
                    'query': query2,
                    'region': region_name,
                    'index': index,
                }
            )

        if len(self.samples) == 0:
            raise RuntimeError(
                'No valid training pairs found. '
                'Please check thermal JSON and region metadata.'
            )

        logging.info(
            f'[{split}] loaded {len(self.samples)} thermal samples; '
            f'positive is cropped from large satellite map (baseline-consistent).'
        )

        if args.max_train_samples > 0 and split in {'train', 'extended'}:
            self.samples = self.samples[: args.max_train_samples]

    def __getstate__(self):
        state = self.__dict__.copy()
        state['region_runtime_cache'] = {}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if 'region_runtime_cache' not in self.__dict__:
            self.region_runtime_cache = {}

    def _get_region_runtime(self, region_name: str):
        if region_name in self.region_runtime_cache:
            return self.region_runtime_cache[region_name]

        meta = self.region_meta[region_name]
        try:
            ref_img = tifffile.memmap(meta['ref_path'])
        except Exception:
            ref_img = tifffile.imread(meta['ref_path'])

        runtime = {
            'config': meta['config'],
            'ref_map_full': ref_img,
        }
        self.region_runtime_cache[region_name] = runtime
        return runtime

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        query_img = Image.open(sample['query']['abs_path']).convert('RGB')
        query_img = _center_square_crop_pil(query_img)
        if bool(getattr(self.args, 'align_query_with_yaw', True)):
            rotation_prior_deg = _extract_rotation_prior_deg(
                sample['query'],
                default_deg=float(getattr(self.args, 'query_rotation_default_deg', 0.0))
            )
            rotation_sign = float(getattr(self.args, 'query_rotation_sign', -1.0))
            query_img = _rotate_query_with_inscribed_crop(
                query_img,
                degree=rotation_sign * rotation_prior_deg,
                keep_size=True,
            )
        region_name = sample['region']
        res = self._get_region_runtime(region_name)
        region_cfg = res['config']

        ref_crop, _, new_cfg = _crop_geo_images_with_offset_same_as_baseline(
            region=region_name,
            config=region_cfg,
            ref_type=self.ref_type,
            crop_size_m=self.args.crop_size_m,
            offset_range_m=self.args.offset_range_m,
            true_pos=sample['query'],
            ref_map_full=res['ref_map_full'],
        )

        positive_img = _sat_crop_to_tensor(ref_crop, self.sat_size_px)

        gt_lon, gt_lat = _get_ground_target_lonlat(sample['query'])
        qx, qy = _deg2utm(region_name, region_cfg, gt_lon, gt_lat)
        ref_res = new_cfg[f'{region_name}_{self.ref_type}_REF_resolution']
        new_initial_x = new_cfg[f'{region_name}_{self.ref_type}_REF_initialX']
        new_initial_y = new_cfg[f'{region_name}_{self.ref_type}_REF_initialY']

        crop_h, crop_w = ref_crop.shape[:2]
        query_col = (qx - new_initial_x) / ref_res
        query_row = (new_initial_y - qy) / ref_res


        DEBUG_VIS = False
        if DEBUG_VIS:

            vis_dir = os.path.join(str(self.root_dir), "debug_vis")
            os.makedirs(vis_dir, exist_ok=True)

            query_img.save(os.path.join(vis_dir, f"{index}_query_thermal.jpg"))
            

            vis_crop = ref_crop.copy()
            
            if vis_crop.ndim == 3 and vis_crop.shape[2] == 3:
                vis_crop = cv2.cvtColor(vis_crop, cv2.COLOR_RGB2BGR)
                
            gt_x, gt_y = int(query_col), int(query_row)
            
            cv2.drawMarker(vis_crop, (gt_x, gt_y), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=40, thickness=3)
            cv2.circle(vis_crop, (gt_x, gt_y), 6, (0, 255, 0), -1)
            
            cv2.imwrite(os.path.join(vis_dir, f"{index}_ref_crop_GT.jpg"), vis_crop)

        scale_x = float(self.sat_size_px) / float(crop_w)
        scale_y = float(self.sat_size_px) / float(crop_h)
        query_col_resized = float(query_col) * scale_x
        query_row_resized = float(query_row) * scale_y
        in_bounds = (
            0.0 <= query_col_resized < float(self.sat_size_px)
            and 0.0 <= query_row_resized < float(self.sat_size_px)
        )
        query_col_resized = float(np.clip(query_col_resized, 0.0, float(self.sat_size_px - 1)))
        query_row_resized = float(np.clip(query_row_resized, 0.0, float(self.sat_size_px - 1)))
        on_black = _is_black_area_on_tensor(positive_img, query_col_resized, query_row_resized)
        valid_gt = torch.tensor(1.0 if (in_bounds and (not on_black)) else 0.0, dtype=torch.float32)
        center_col_resized = (float(self.sat_size_px) - 1.0) / 2.0
        center_row_resized = (float(self.sat_size_px) - 1.0) / 2.0
        
        query_utm = torch.from_numpy(np.array([[query_row_resized, query_col_resized]], dtype=np.float32))
        positive_utm = torch.from_numpy(np.array([[center_row_resized, center_col_resized]], dtype=np.float32))

        batch = super(BenchmarkJsonDataset, self).__getitem__(
            query_img,
            positive_img,
            query_utm,
            positive_utm,
            index,
            index,
        )
        return (*batch, valid_gt)


def fetch_dataloader(args, split='train'):
    ds = BenchmarkJsonDataset(args=args, split=split)
    common_kwargs = {
        'pin_memory': True,
        'num_workers': args.num_workers,
        'worker_init_fn': seed_worker,
    }
    if args.num_workers > 0:
        common_kwargs['persistent_workers'] = True
        if os.name == 'nt':
            common_kwargs['multiprocessing_context'] = 'spawn'

    if split in {'train', 'extended'}:
        loader = data.DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            **common_kwargs,
        )
    else:
        generator = torch.Generator()
        generator.manual_seed(0)
        loader = data.DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            generator=generator,
            **common_kwargs,
        )
    return loader
