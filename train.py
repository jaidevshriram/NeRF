import os
import time

import wandb
from cprint import cprint
import argparse

from configs import get_cfg_defaults

import torch
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything

from datasets import make_datamodule
from trainer import NeRF

from pytorch_lightning.profilers import AdvancedProfiler

from utils.logging import init_logger

def get_intrinsics(dm: pl.LightningDataModule, stage: str='fit'):
    dataloader = {
        'fit': dm.train_dataloader,
        'eval': dm.val_dataloader,
        'test': dm.test_dataloader,
    }
    
    dm.setup(stage)
    it = iter(dataloader[stage]())
    batch = next(it)

    return batch['intrinsics']


def main(cfg):
    # Make datamodule
    dm = make_datamodule(cfg)
    cprint.warn(f"Datamodule made")

    # Get Batched Intrinsics
    train_intrinsics = get_intrinsics(dm=dm, stage='fit')
    eval_intrinsics = get_intrinsics(dm=dm, stage='eval')

    # Make model
    model = NeRF(cfg.learning_rate, (cfg.width, cfg.height), cfg=dict(cfg), train_intrinsics=train_intrinsics, eval_intrinsics=eval_intrinsics)
    device = "cpu" if cfg.no_cuda else "gpu"
    cprint.warn(f"Model NeRF made, running on " + device)

    # Unique timstamp for logging:
    ts = time.strftime("%Y%m%d-%H%M%S")

    # Prepare Logging
    log_dir = os.path.join(cfg.log_dir, cfg.experiment_name, ts)
    os.makedirs(log_dir)
    logger = WandbLogger(name=cfg.experiment_name, project=cfg.project_name, save_dir=log_dir, log_model=cfg.log_model_checkpoint)
    # logger.watch(model, log='all')

    # Prepare checkpointing and saving
    weight_dir = os.path.join(cfg.weight_dir, cfg.experiment_name, ts)
    os.makedirs(weight_dir)
    checkpoint_callback = ModelCheckpoint(
        dirpath=weight_dir,
        monitor='val_loss',
        filename= cfg.experiment_name + '-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
        every_n_epochs=1
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # aprofiler = AdvancedProfiler(filename='perf_logs')

    # model = NeRF.load_from_checkpoint("weights/misty-paper-42/20221204-180935/misty-paper-42-epoch=01-val_loss=0.47.ckpt")

    # Train!
    if device == 'gpu':
        trainer = pl.Trainer(accelerator=device, devices=torch.cuda.device_count(), logger=logger,
                callbacks=[checkpoint_callback, lr_monitor], check_val_every_n_epoch=1, max_epochs=cfg.num_epochs)
    else:
        trainer = pl.Trainer(accelerator=device, logger=logger,
                callbacks=[checkpoint_callback, lr_monitor], check_val_every_n_epoch=1, max_epochs=cfg.num_epochs)
    trainer.fit(model, dm)

    # Test!
    trainer.test(model, dm)

if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument("--experiment_name", help="experiment name", required=False, default="")
    args.add_argument("--config_path", help="Path to Config File", required=False, default="")
    args, _ = args.parse_known_args()
    
    # Load config file
    cfg = get_cfg_defaults()

    if os.path.exists(args.config_path) and os.path.splitext(args.config_path)[1] == '.yaml':
        cfg.merge_from_file(args.config_path)
    else:
        print("No valid config specified, using  default")

    if args.experiment_name != "":
        cfg.update({'experiment_name': args.experiment_name}, allow_val_change=True)
    
    cfg = init_logger('wandb', cfg)
    cprint.info(cfg)

    main(cfg)