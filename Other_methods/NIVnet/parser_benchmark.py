import argparse
from pathlib import Path


def _str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {'1', 'true', 't', 'yes', 'y', 'on'}:
        return True
    if value in {'0', 'false', 'f', 'no', 'n', 'off'}:
        return False
    raise argparse.ArgumentTypeError(f'Invalid boolean value: {value}')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='STHN-Benchmark-Finetune', help='experiment name')
    parser.add_argument('--restore_ckpt', type=str, default=None, help='checkpoint path for finetuning')
    parser.add_argument('--gpuid', type=int, nargs='+', default=[0])
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])

    parser.add_argument('--lev0', default=True, action='store_true', help='coarse level')
    parser.add_argument('--iters_lev0', type=int, default=6)
    parser.add_argument('--iters_lev1', type=int, default=6)
    parser.add_argument('--two_stages', action='store_true')
    parser.add_argument('--corr_level', type=int, default=2, choices=[2, 4, 6])
    parser.add_argument('--arch', type=str, default='IHN', choices=['IHN'])
    parser.add_argument('--fnet_cat', action='store_true')
    parser.add_argument('--fine_padding', type=int, default=0)
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--augment_two_stages', type=float, default=0)

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_scheduler', type=str, default='warmup_cosine',
                        choices=['warmup_cosine', 'onecycle', 'cosine', 'step'],
                        help='dynamic learning rate schedule type')
    parser.add_argument('--warmup_steps', type=int, default=100,
                        help='warmup steps for warmup_cosine scheduler')
    parser.add_argument('--min_lr_ratio', type=float, default=0.05,
                        help='minimum learning rate ratio relative to base lr')
    parser.add_argument('--onecycle_pct_start', type=float, default=0.1,
                        help='percentage of steps spent increasing lr in onecycle scheduler')
    parser.add_argument('--onecycle_div_factor', type=float, default=10.0,
                        help='initial lr = max_lr / div_factor for onecycle scheduler')
    parser.add_argument('--onecycle_final_div_factor', type=float, default=100.0,
                        help='minimum lr = initial lr / final_div_factor for onecycle scheduler')
    parser.add_argument('--num_steps', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--wdecay', type=float, default=1e-5)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.85)
    parser.add_argument('--mixed_precision', default=False, action='store_true')
    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument('--G_loss_lambda', type=float, default=1.0)
    parser.add_argument('--database_size', type=int, default=512, choices=[512, 1024, 1536], help='deprecated in benchmark mode; overridden by crop_size_m')
    parser.add_argument('--resize_width', type=int, default=256, choices=[256, 512])
    parser.add_argument('--crop_width', type=int, default=512, choices=[256, 512])
    parser.add_argument('--augment', type=str, default='none', choices=['none', 'img', 'ue'])
    parser.add_argument('--rotate_max', type=float, default=0)
    parser.add_argument('--resize_max', type=float, default=0)
    parser.add_argument('--perspective_max', type=float, default=0)
    parser.add_argument('--vis_all', action='store_true')
    parser.add_argument('--weight', action='store_true')
    parser.add_argument('--finetune', action='store_true')

    default_root = Path(__file__).resolve().parents[2]
    parser.add_argument('--root_dir', type=str, default=str(default_root), help='Benchmark-main root path')
    parser.add_argument('--global_config_yaml', type=str, default='config.yaml', help='global benchmark config yaml')
    parser.add_argument('--region_yaml_dir', type=str, default='Regions_params', help='region yaml directory')
    parser.add_argument('--Ref_type', type=str, default='HIGH', choices=['HIGH', 'LOW'])
    parser.add_argument('--crop_size_m', type=float, default=1200.0, help='satellite crop size in meters')
    parser.add_argument('--offset_range_m', type=float, default=120.0, help='random crop center offset range in meters')
    parser.add_argument('--align_query_with_yaw', type=_str2bool, nargs='?', const=True, default=True,
                        help='rotate query image by yaw prior and crop with inscribed-circle strategy')
    parser.add_argument('--query_rotation_sign', type=float, default=-1.0, help='rotation direction sign for query alignment, usually -1 or +1')
    parser.add_argument('--query_rotation_default_deg', type=float, default=0.0, help='fallback degree when rotation prior is missing')
    parser.add_argument('--use_query_center_crop', type=_str2bool, nargs='?', const=True, default=False,
                        help='whether to apply center crop (crop_width) on query image inside homo_dataset')
    parser.add_argument('--niv_base_channels', type=int, default=24, help='base channel width for NIVnet')
    parser.add_argument('--niv_res_blocks', type=int, default=3, help='number of residual blocks in encoder/generator')
    parser.add_argument('--niv_attr_dim', type=int, default=16, help='attribute bottleneck dimension')
    parser.add_argument('--niv_match_hw', type=int, default=16, help='spatial size for correlation matching feature maps')
    parser.add_argument('--niv_match_temperature', type=float, default=0.07, help='temperature in softmax matching')
    parser.add_argument('--niv_reg_hidden_channels', type=int, default=192, help='regression head stage-1 channels')
    parser.add_argument('--niv_reg_mid_channels', type=int, default=96, help='regression head stage-2 channels')
    parser.add_argument('--niv_reg_low_channels', type=int, default=48, help='regression head stage-3 channels')
    parser.add_argument('--niv_reg_dropout', type=float, default=0.1, help='dropout in regression head')
    parser.add_argument('--niv_use_channel_attention', type=_str2bool, nargs='?', const=True, default=True,
                        help='enable lightweight channel attention in residual blocks')
    parser.add_argument('--max_params', type=int, default=10_000_000, help='hard cap for trainable parameters')

    parser.add_argument('--lambda_recon', type=float, default=10.0, help='weight of reconstruction loss')
    parser.add_argument('--lambda_trans', type=float, default=1.0, help='weight of translation consistency loss')
    parser.add_argument('--lambda_grid', type=float, default=10.0, help='weight of grid alignment loss')
    parser.add_argument('--lambda_inten', type=float, default=1.0, help='weight of intensity consistency loss')
    parser.add_argument('--lambda_iden', type=float, default=5.0, help='weight of invertibility loss')

    parser.add_argument('--train_thermal_json', type=str, default='Data/metadata/train_Thermal.json')
    parser.add_argument('--val_thermal_json', type=str, default='Data/metadata/valid_Thermal.json')
    parser.add_argument('--test_thermal_json', type=str, default='Data/metadata/test_Thermal.json')
    parser.add_argument('--max_train_samples', type=int, default=-1)

    parser.add_argument('--save_root', type=str, default='logs/local_benchmark')
    parser.add_argument('--save_freq', type=int, default=2000)
    parser.add_argument('--print_freq', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--use_swanlab', default=True, help='enable SwanLab tracking')
    parser.add_argument('--swanlab_project', type=str, default='NIVnet')
    parser.add_argument('--swanlab_experiment', type=str, default=None)

    parser.add_argument('--enable_train_vis', default=False, help='save periodic batch visualization')
    parser.add_argument('--vis_interval', type=int, default=1, help='save visualization every N steps')
    parser.add_argument('--vis_max_items', type=int, default=1, help='max samples to render per batch image')

    parser.add_argument('--run_val', type=_str2bool, nargs='?', const=True, default=True, help='run validation during training')
    parser.add_argument('--run_test', type=_str2bool, nargs='?', const=True, default=True, help='run test after training ends')
    parser.add_argument('--eval_val_every', type=int, default=1000, help='evaluate validation every N training steps')
    parser.add_argument('--eval_max_batches', type=int, default=-1, help='max batches for val/test eval; -1 means full set')

    args = parser.parse_args()
    args.sat_size_px = max(2, int(round(float(args.crop_size_m))))
    args.database_size = args.sat_size_px
    return args
