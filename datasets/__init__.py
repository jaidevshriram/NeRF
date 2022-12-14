from .nerf import NerfDataModule

def make_datamodule(cfg):
    dm = NerfDataModule(cfg.data_dir, cfg.split_dir, (cfg.width, cfg.height), cfg.batch_size, cfg.num_workers)
    return dm