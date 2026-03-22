import argparse
import math
import os
import pprint
from distutils.util import strtobool
from pathlib import Path

import matplotlib
import pytorch_lightning as pl
from loguru import logger as loguru_logger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.utilities import rank_zero_only
from src.config.default import get_cfg_defaults
from src.lightning.data import MultiSceneDataModule
from src.lightning.lightning_loftr import PL_LoFTR
from src.utils.misc import get_rank_zero_only_logger, setup_gpus
from src.utils.profiler import build_profiler

matplotlib.use('Agg')

loguru_logger = get_rank_zero_only_logger(loguru_logger)


class SaveCheckpointEveryNBatch(Callback):
    def __init__(self, save_dir, every_n_batch=100, prefix="batch"):
        super().__init__()
        self.save_dir = save_dir
        self.every_n_batch = every_n_batch
        self.prefix = prefix

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if trainer.is_global_zero and (batch_idx + 1) % self.every_n_batch == 0:
            filename = os.path.join(self.save_dir, f"{self.prefix}_epoch{trainer.current_epoch}_batch{batch_idx}.ckpt")
            trainer.save_checkpoint(filename)
            print(f"Checkpoint saved at {filename}")


def parse_args():
    # init a costum parser which will be added into pl.Trainer parser
    # check documentation: https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'data_cfg_path', type=str, help='data config path')
    parser.add_argument(
        'main_cfg_path', type=str, help='main config path')
    parser.add_argument(
        '--exp_name', type=str, default='default_exp_name')
    parser.add_argument(
        '--batch_size', type=int, default=4, help='batch_size per gpu')
    parser.add_argument(
        '--num_workers', type=int, default=4)
    parser.add_argument(
        '--pin_memory', type=lambda x: bool(strtobool(x)),
        nargs='?', default=True, help='whether loading data to pinned memory or not')
    parser.add_argument(
        '--ckpt_path', type=str, default=None,
        help='pretrained checkpoint path, helpful for using a pre-trained coarse-only LoFTR')
    parser.add_argument(
        '--disable_ckpt', action='store_true',
        help='disable checkpoint saving (useful for debugging).')
    parser.add_argument(
        '--profiler_name', type=str, default=None,
        help='options: [inference, pytorch], or leave it unset')
    parser.add_argument(
        '--parallel_load_data', action='store_true',
        help='load datasets in with multiple processes.')
    parser.add_argument(
        '--save_top_k', type=int, default=10,
        help='save top k checkpoints based on the monitored metric')
    parser.add_argument(
        '--lr_scale', type=float, default=1.0,
        help='scale the learning rate for fine-tuning')
    parser.add_argument(
        '--modality_list', nargs='+', default=['visible'],
        help='List of modalities')
    parser.add_argument(
        '--save_dir', type=str, default=None,
        help='save directory')

    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args()


def main():
    # parse arguments
    args = parse_args()
    rank_zero_only(pprint.pprint)(vars(args))

    # init default-cfg and merge it with the main- and data-cfg
    config = get_cfg_defaults()
    config.merge_from_file(args.main_cfg_path)
    config.merge_from_file(args.data_cfg_path)
    pl.seed_everything(config.TRAINER.SEED)  # reproducibility
    # TODO: Use different seeds for each dataloader workers
    # This is needed for data augmentation
    # print('modality_list:', args.modality_list)

    # scale lr and warmup-step automatically
    args.gpus = _n_gpus = setup_gpus(args.gpus)
    config.TRAINER.WORLD_SIZE = _n_gpus * args.num_nodes
    config.TRAINER.TRUE_BATCH_SIZE = config.TRAINER.WORLD_SIZE * args.batch_size
    _scaling = config.TRAINER.TRUE_BATCH_SIZE / config.TRAINER.CANONICAL_BS
    config.TRAINER.SCALING = _scaling
    config.TRAINER.TRUE_LR = config.TRAINER.CANONICAL_LR * _scaling
    lr_scale = args.lr_scale
    config.TRAINER.TRUE_LR = config.TRAINER.TRUE_LR * lr_scale  # for fine-tuning
    config.TRAINER.WARMUP_STEP = math.floor(config.TRAINER.WARMUP_STEP / _scaling)

    # lightning module
    profiler = build_profiler(args.profiler_name)
    model = PL_LoFTR(config, pretrained_ckpt=args.ckpt_path, profiler=profiler)
    loguru_logger.info(f"LoFTR LightningModule initialized!")

    # lightning data
    data_module = MultiSceneDataModule(args, config)
    loguru_logger.info(f"LoFTR DataModule initialized!")

    # TensorBoard Logger
    save_dir = Path(args.save_dir)
    logger = TensorBoardLogger(save_dir=f'{save_dir}/logs/tb_logs', name=args.exp_name, default_hp_metric=False)
    ckpt_dir = Path(logger.log_dir) / 'checkpoints'
    save_config_dir = Path(logger.log_dir) / 'configs'
    save_config_dir.mkdir(parents=True, exist_ok=True)
    with open(save_config_dir / 'config.yaml', 'w') as f:
        f.write(config.dump())

    # Callbacks
    # TODO: update ModelCheckpoint to monitor multiple metrics
    batch_ckpt_callback = SaveCheckpointEveryNBatch(save_dir=str(ckpt_dir), every_n_batch=1000)
    ckpt_callback = ModelCheckpoint(monitor='auc@10', verbose=True, save_top_k=args.save_top_k, mode='max',
                                    save_last=False,
                                    dirpath=str(ckpt_dir),
                                    filename='{epoch}-{auc@5:.3f}-{auc@10:.3f}-{auc@20:.3f}')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [lr_monitor]
    if not args.disable_ckpt:
        callbacks.append(ckpt_callback)

    # Lightning Trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        resume_from_checkpoint=args.resume_from_checkpoint,
        plugins=DDPPlugin(find_unused_parameters=False,
                          num_nodes=args.num_nodes,
                          sync_batchnorm=config.TRAINER.WORLD_SIZE > 0),
        gradient_clip_val=config.TRAINER.GRADIENT_CLIPPING,
        callbacks=callbacks,
        logger=logger,
        sync_batchnorm=config.TRAINER.WORLD_SIZE > 0,
        replace_sampler_ddp=False,  # use custom sampler
        reload_dataloaders_every_epoch=False,  # avoid repeated samples!
        weights_summary='full',
        profiler=profiler)
    loguru_logger.info(f"Trainer initialized!")
    loguru_logger.info(f"Start training!")
    trainer.fit(model, datamodule=data_module)


if __name__ == '__main__':
    main()
