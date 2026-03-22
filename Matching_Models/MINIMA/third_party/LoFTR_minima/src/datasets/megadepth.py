import os
import os.path as osp
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from loguru import logger
from src.utils.dataset import read_megadepth_gray, read_megadepth_depth
from torch.utils.data import Dataset


class MegaDepthDataset(Dataset):
    def __init__(self,
                 root_dir,
                 npz_path,
                 mode='train',
                 min_overlap_score=0.4,
                 img_resize=None,
                 df=None,
                 img_padding=False,
                 depth_padding=False,
                 augment_fn=None,
                 modality_list=None,
                 **kwargs):
        """
        Manage one scene(npz_path) of MegaDepth dataset.

        Args:
            root_dir (str): megadepth root directory that has `phoenix`.
            npz_path (str): {scene_id}.npz path. This contains image pair information of a scene.
            mode (str): options are ['train', 'val', 'test']
            min_overlap_score (float): how much a pair should have in common. In range of [0, 1]. Set to 0 when testing.
            img_resize (int, optional): the longer edge of resized images. None for no resize. 640 is recommended.
                                        This is useful during training with batches and testing with memory intensive algorithms.
            df (int, optional): image size division factor. NOTE: this will change the final image size after img_resize.
            img_padding (bool): If set to 'True', zero-pad the image to squared size. This is useful during training.
            depth_padding (bool): If set to 'True', zero-pad depthmap to (2000, 2000). This is useful during training.
            augment_fn (callable, optional): augments images with pre-defined visual effects.
        """
        super().__init__()
        if modality_list is None:
            self.modal_options = ['visible']
        else:
            self.modal_options = modality_list
        self.infrared_root = 'data/megadepth/train/infrared/'
        self.depth_root = 'data/megadepth/train/depth/'
        self.normal_root = 'data/megadepth/train/normal/'
        self.paint_root = 'data/megadepth/train/paint/'
        self.sketch_root = 'data/megadepth1500/sketch/'
        self.event_root = 'data/megadepth/train/event/'
        self.root_dir = root_dir
        self.mode = mode
        self.scene_id = npz_path.split('.')[0]

        # print('min_overlap_score', min_overlap_score)

        # prepare scene_info and pair_info
        if mode == 'test' and min_overlap_score != 0:
            logger.warning("You are using `min_overlap_score`!=0 in test mode. Set to 0.")
            min_overlap_score = 0
        self.scene_info = np.load(npz_path, allow_pickle=True)
        self.pair_infos = self.scene_info['pair_infos'].copy()
        self.scene_info = dict(self.scene_info)
        del self.scene_info['pair_infos']
        self.pair_infos = [pair_info for pair_info in self.pair_infos if pair_info[1] > min_overlap_score]

        # parameters for image resizing, padding and depthmap padding
        # if mode == 'train':
        # assert img_resize is not None and img_padding and depth_padding
        self.img_resize = img_resize
        self.df = df
        self.img_padding = img_padding
        self.depth_max_size = 2000 if depth_padding else None  # the upperbound of depthmaps size in megadepth.

        # for training LoFTR
        self.augment_fn = augment_fn if mode == 'train' else None
        self.coarse_scale = getattr(kwargs, 'coarse_scale', 0.125)

        if self.mode == 'val':
            self.dataset_length = len(self.pair_infos)
        else:
            self.dataset_length = len(self.pair_infos) * 2
        self.local_random_1 = random.Random(time.time())
        self.local_random_2 = random.Random(time.time() + 666)

        self.modality_to_root = {
            'visible': self.root_dir,
            'infrared': self.infrared_root,
            'depth': self.depth_root,
            'normal': self.normal_root,
            'event': self.event_root,
            'sketch': self.sketch_root,
            'paint': self.paint_root,
        }

        if self.mode == 'val':
            self.modal_options = ['visible']

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        original_idx = idx // 2
        reverse_mode = self.local_random_1.randint(0, 1)  # 0: visible-infrared, 1: infrared-visible

        (idx0, idx1), overlap_score, central_matches = self.pair_infos[original_idx]

        # read grayscale image and mask. (1, h, w) and (h, w)
        img_name0 = osp.join(self.root_dir, self.scene_info['image_paths'][idx0])
        img_name1 = osp.join(self.root_dir, self.scene_info['image_paths'][idx1])

        if self.mode == 'train' or self.mode == 'val':
            self.multi_model = self.local_random_2.choice(self.modal_options)
            self.multi_model = str(self.multi_model).strip().strip('"').strip("'")
            try:
                self.multi_model_root = self.modality_to_root[self.multi_model]
            except KeyError:
                raise ValueError(f"Unknown multi_model: {self.multi_model}")

            # print('self.multi_model', self.multi_model, 'reverse_mode', reverse_mode)

            img_name0_x_modality = osp.join(self.multi_model_root, self.scene_info['image_paths'][idx0])
            img_name1_x_modality = osp.join(self.multi_model_root, self.scene_info['image_paths'][idx1])

            if reverse_mode:

                if self.multi_model in ['infrared', 'depth', 'normal', 'sketch']:
                    filename_wo_ext, _ = os.path.splitext(img_name0_x_modality)
                    img_name0 = filename_wo_ext + '.jpg'
                    # assert os.path.exists(img_name1), f"img not found: {img_name1}"
                    # assert os.path.exists(img_name0_x_modality), f"img not found: {img_name0_x_modality}"

                elif self.multi_model in ['event']:

                    filename_wo_ext, _ = os.path.splitext(img_name0_x_modality)
                    img_name0 = filename_wo_ext + '.png'
                    # assert os.path.exists(img_name1), f"img not found: {img_name1}"
                    # assert os.path.exists(img_name0_x_modality), f"img not found: {img_name0_x_modality}"
                elif self.multi_model == 'visible':
                    pass
                else:
                    raise ValueError(f"Unknown multi_model: {self.multi_model}")

            else:
                if self.multi_model in ['infrared', 'depth', 'normal', 'sketch']:
                    filename_wo_ext, _ = os.path.splitext(img_name1_x_modality)
                    img_name1 = filename_wo_ext + '.jpg'

                    # assert os.path.exists(img_name0), f"img not found: {img_name0}"
                    # assert os.path.exists(img_name1_x_modality), f"img not found: {img_name1_x_modality}"

                elif self.multi_model in ['event']:
                    filename_wo_ext, _ = os.path.splitext(img_name1_x_modality)
                    img_name1 = filename_wo_ext + '.png'
                    # assert os.path.exists(img_name0), f"img not found: {img_name0}"
                    # assert os.path.exists(img_name1_x_modality), f"img not found: {img_name1_x_modality}"

                elif self.multi_model == 'visible':
                    pass

                else:
                    raise ValueError(f"Unknown multi_model: {self.multi_model}")

            image0, mask0, scale0 = read_megadepth_gray(img_name0, self.img_resize, self.df, self.img_padding,
                                                        None)

            image1, mask1, scale1 = read_megadepth_gray(img_name1, self.img_resize, self.df,
                                                        self.img_padding,
                                                        None)

        else:
            img_name0_x_modality = osp.join(self.multi_model_root, self.scene_info['image_paths'][idx0])
            img_name1_x_modality = osp.join(self.multi_model_root, self.scene_info['image_paths'][idx1])
            if reverse_mode:
                img_name0_x_modality = img_name0_x_modality.replace('jpg', 'png')

                image0, mask0, scale0 = read_megadepth_gray(img_name0_x_modality, self.img_resize, self.df,
                                                            self.img_padding,
                                                            None)
                image1, mask1, scale1 = read_megadepth_gray(img_name1, self.img_resize, self.df, self.img_padding,
                                                            None)
                img_name0 = img_name0_x_modality
            else:
                img_name1_x_modality = img_name1_x_modality.replace('jpg', 'png')

                image0, mask0, scale0 = read_megadepth_gray(img_name0, self.img_resize, self.df, self.img_padding,
                                                            None)
                image1, mask1, scale1 = read_megadepth_gray(img_name1_x_modality, self.img_resize, self.df,
                                                            self.img_padding,
                                                            None)
                img_name1 = img_name1_x_modality

        # read depth. shape: (h, w)
        if self.mode in ['train', 'val']:
            depth0 = read_megadepth_depth(
                osp.join(self.root_dir, self.scene_info['depth_paths'][idx0]), pad_to=self.depth_max_size)
            depth1 = read_megadepth_depth(
                osp.join(self.root_dir, self.scene_info['depth_paths'][idx1]), pad_to=self.depth_max_size)
        else:
            depth0 = depth1 = torch.tensor([])

        # read intrinsics of original size
        K_0 = torch.tensor(self.scene_info['intrinsics'][idx0].copy(), dtype=torch.float).reshape(3, 3)
        K_1 = torch.tensor(self.scene_info['intrinsics'][idx1].copy(), dtype=torch.float).reshape(3, 3)

        # read and compute relative poses
        T0 = self.scene_info['poses'][idx0]
        T1 = self.scene_info['poses'][idx1]
        T_0to1 = torch.tensor(np.matmul(T1, np.linalg.inv(T0)), dtype=torch.float)[:4, :4]  # (4, 4)
        T_1to0 = T_0to1.inverse()
        # print('img_name0', img_name0)
        # print('img_name1', img_name1)

        data = {
            'image0': image0,  # (1, h, w)
            'depth0': depth0,  # (h, w)
            'image1': image1,
            'depth1': depth1,
            'T_0to1': T_0to1,  # (4, 4)
            'T_1to0': T_1to0,
            'K0': K_0,  # (3, 3)
            'K1': K_1,
            'scale0': scale0,  # [scale_w, scale_h]
            'scale1': scale1,
            'dataset_name': 'MegaDepth',
            'scene_id': self.scene_id,
            'pair_id': idx,
            'pair_names': (img_name0, img_name1),
        }

        # for LoFTR training
        if mask0 is not None:  # img_padding is True
            if self.coarse_scale:
                [ts_mask_0, ts_mask_1] = F.interpolate(torch.stack([mask0, mask1], dim=0)[None].float(),
                                                       scale_factor=self.coarse_scale,
                                                       mode='nearest',
                                                       recompute_scale_factor=False)[0].bool()
            data.update({'mask0': ts_mask_0, 'mask1': ts_mask_1})

        return data
