<div align="center">
  <img src="assets/UAV_log.png" alt="logo head" width="150"/>
  <br><br>
  <h1>SCC-Loc: A Unified Semantic Cascade Consensus Framework for UAV Thermal Geo-Localization</h1>
  <a href="http://arxiv.org/abs/2604.03120"><img src="https://img.shields.io/badge/arXiv-2604.03120-b31b1b.svg" alt="arXiv"></a>
  <a href="assets/video.mp4"><img src="https://img.shields.io/badge/Video-Demo-FF0000?logo=youtube&logoColor=white" alt="Video"></a>
  <a href="https://github.com/FloralHercules/SCC-Loc"><img src="https://img.shields.io/github/stars/FloralHercules/SCC-Loc?style=social" alt="Stars"></a>
  <p>
    If you find our work useful, please consider giving us a ⭐️. 
    Your support means a lot to us! 🥰
  </p>
</div>

## Framework

</p>
<div align="center">
  <img src="assets/Pipeline.png" alt="Logo" width="1000"/>
</div>

🎯 **SCC-Loc** is a zero-shot, cross-modal thermal geo-localization framework for UAVs operating in GNSS-denied environments. Matching onboard thermal images with visible-light satellite maps is challenging due to the massive modality gap. SCC-Loc tackles this by providing highly accurate absolute pose estimation  **without needing domain-specific training** .

**✨ Key Highlights:**

* **Zero-Shot Capability:** Powered by a shared DINOv2 backbone, requiring no specific retraining for new environments.
* **Robust Matching:** Overcomes dense structural outliers and visual decoys.
* **Unified Framework:** Integrates Semantic-Guided Viewport Alignment, Spatial-Adaptive Filtering, and Consensus-Driven Position Selection.

## Dataset

<div align="center">
  <img src="assets/Dataset.png" alt="Logo" width="1000"/>
</div>

🀄 We constructed the **Thermal-UAV** dataset, comprising 11,890 thermal UAV images captured using a DJI Matrice 4T drone around Changsha. These images feature nadir views, multi-temporal and multi-scenario data, and continuous flight sequences. We split the dataset into three sets: 8,115 for training, 1,425 for validation, and 2,350 for testing. Additionally, we collected corresponding Google Maps satellite imagery at 0.26 m/px and a Digital Surface Model (DSM) at 5.29 m/px. The directory structure of the Thermal-UAV dataset is as follows:

```text
Data/
├── metadata/
│   ├── train_Thermal.json
│   ├── valid_Thermal.json
│   └── test_Thermal.json
├── Reference_map/
│   └── changsha/
│       ├── ref.tif
│       └── dsm.tif
└── Thermal-UAV/
  ├── train/
  │   └── changsha/
  │       └── <place_name>/
  │           └── Thermal/
  │               ├── xxx1.JPG
  │               ├── xxx2.JPG
  |		  └── ...
  |	      └──  Thermal_info.csv
  ├── valid/
  └── test/
```

The Thermal-UAV dataset and checkpoint are provided in [Baidu Netdisk](https://pan.baidu.com/s/1zXn3f9QrO07IcKHr8kmNnQ?pwd=53ea) and [Hugging Face](https://huggingface.co/datasets/FloralHercules/Thermal-UAV/tree/main). You can use the `process.ipynb` to get the metadate of Thermal-UAV JSON format, as follows:

```json
{
  "name": "./Data/Thermal-UAV/train/changsha/city_300_ortho_night/Thermal/xxx_T.JPG",
  "lat": 28.2436611, # latitude
  "lon": 112.9985009, # longitude
  "abs_height": 341.354, # absolute height
  "rel_height": 323.967, # relative take-off point altitude
  "pitch": -90.0, # pitch
  "yaw": 88.0, # yaw
  "roll": 180.0, # roll
  "cam_size": 9.83, # diagonal physical size of the sensor
  "focal_len": 12.0, # focal length (mm)
  "width": 1280.0, # image width
  "height": 1024.0 # image height
}
```

It is noted that we need to fill in the corresponding changsha geographic infomation in Regions_params/, as follows:

```yaml
changsha_UTM_SYSTEM: 49N
changsha_SAMPLE_INTERVAL: 10
changsha_HIGH_REF_PATH: ./Data/Reference_map/changsha/ref.tif
changsha_HIGH_DSM_PATH: ./Data/Reference_map/changsha/dsm.tif
changsha_HIGH_REF_initialX: 694811.4577 # UTM coordinate
changsha_HIGH_REF_initialY: 3130673.6615 # UTM coordinate
changsha_HIGH_REF_resolution: 0.3 # Align the resolution through upsampling on QGIS
changsha_HIGH_DSM_resolution: 0.3 # Align the resolution through upsampling on QGIS
changsha_HIGH_REF_COORDINATE: # If the satellite image and DSM are aligned. Their offset is 0
  - 0.0
  - 0.0
changsha_HIGH_DSM_COORDINATE: 
  - 0.0
  - 0.0
```

## Checkpoints

### 1. Retrieval Model

* **CAMP**

```text
Retrieval_Models/
└── CAMP/
  └── weights/
    └── weights_0.9446_for_U1652.pth
```

* **DINOv3** and **DINOv2** are offered through torch.hub. They will automatically download if  your internet connection is available .

### 2. Matching Model

* **RoMa**

```text
Matching_Models/
└── RoMa/
    └── ckpt/
        ├── roma_outdoor.pth
        └── dinov2_vitl14_pretrain.pth
```

* **MINIMA**

```text
Matching_Models/
└── MINIMA/
    └── weights/
        ├── minima_eloftr.pth
        ├── minima_roma.pth
        ├── minima_loftr.ckpt
        ├── minima_lightglue.pth
        └── minima_xoftr.ckpt
```

* **RoMaV2** automatically download

## Method

In  `config.yaml`， choose the **MINIMA_Roma_DINOv**2 as retrieval method and  **MINIMA_Roma** as matching method. Our proposed SCC-Loc framework composes this combination. It is noted that when use **MINIMA_Roma_DINOv**2, it must use **MINIMA_Roma**, while the reverse is not true. Other combinations can be the comparision baseline.

Then, enjoy the fun of operation through:

```
python Baseline.py
```

If you want to see the visualization, please set the `SHOW_RETRIEVAL_RESULT=True` in config.yaml. It will show retrieval, matching, final localization resules etc.

🚀 Furthermore , `process.ipynb` provides useful scripts for constructing custom datasets and reproducing experimental results. Given the  current strict regulations and flight restrictions on UAVs , we warmly invite researchers dedicated to thermal geo-localization to contribute to this open-source dataset,  **expanding its coverage to encompass diverse countries, regions, environments, and viewpoints** .

## Citation

If you find this code or our Thermal-UAV dataset useful for your research, please consider citing our paper:

```
@article{sccloc2026,
  title={SCC-Loc: A Unified Semantic Cascade Consensus Framework for UAV Thermal Geo-Localization},
  author={Xiaoran Zhang, Yu Liu, Jinyu Liang, Kangqiushi Li, Zhiwei Huang, Huaxin Xiao},
  journal={arXiv:2604.01581},
  year={2026}
}
```

## License

We use the Apache License 2.0. See detailed information in LICENSE file.

## Acknowledgements

We are grateful for the publicly available resources and open-source libraries that have been instrumental in this work.

* [A Cross-View Geo-Localization Method using Contrastive Attributes Mining and Position-aware Partitioning](https://github.com/Mabel0403/CAMP)
* [DINOv2: Learning Robust Visual Features without Supervision](https://github.com/facebookresearch/dinov2/tree/main)
* [DINOv3](https://github.com/facebookresearch/dinov3)
* [RoMa: Robust Dense Feature Matching](https://github.com/Parskatt/RoMa)
* [RoMa v2: Harder Better Faster Denser Feature Matching](https://github.com/Parskatt/romav2)
* [MINIMA: Modality Invariant Image Matching](https://github.com/LSXI7/MINIMA)

We are especially grateful to Yibin Ye et al. for their seminal benchmark [Exploring the best way for UAV visual localization under Low-altitude Multi-view Observation Condition: a Benchmark](https://github.com/UAV-AVL/Benchmark?tab=readme-ov-file)

Thank you for your open-source spirit, which has significantly accelerated progress in the UAV visual geo-localization community. Salute to you!
