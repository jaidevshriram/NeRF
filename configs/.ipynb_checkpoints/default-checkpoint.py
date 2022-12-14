# Default Configuration values and structure of Configuration Node

from yacs.config import CfgNode as CN

_C = CN()

# Project level configuration
_C.project_name = 'nerf_dummy'

# Log configuration
_C.experiment_name = "pos-10-dir-4-lr1e-4-bigger-model"
_C.log_dir = '/tmp/nerf/logs'
_C.weight_dir = './weights'
_C.log_frequency = 250
_C.save_frequency = 1
_C.log_model_checkpoint = True  # For wandb
_C.script_mode = 'train'  # train, eval, predict

# Dataset configuration
_C.data_dir = 'data/bottles'
_C.split_dir = 'splits/bottles'
_C.width = 100
_C.height = 100

# Model configuration
_C.load_ckpt_path = None
_C.model_hparams = CN(new_allowed=True)   # To allow model specific params

_C.model = 'nerf'
_C.model_hparams.pos_emb_num = 10
_C.model_hparams.dir_emb_num = 4
_C.model_hparams.hidden_dim = 256

_C.raysampler = CN(new_allowed=True)
_C.raysampler.num_rays = 4096
_C.raysampler.num_pts = 64
_C.raysampler.min_depth = 0.1
_C.raysampler.max_depth = 5.0
_C.raysampler.stratified = False

# Training hyperparameters
_C.num_epochs = 4000
_C.no_cuda = False
_C.batch_size = 1
_C.iterations = 20
_C.num_workers = 2
_C.learning_rate = 1e-4
_C.scheduler_step_size = 2500
_C.seed = 215

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()
