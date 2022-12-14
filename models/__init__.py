import torch.nn as nn

from .nerf import NeRFModel
from .tiny_nerf import TinyNeRFModel

from .transformers import PositionalEncoding

models = {
    'tiny-nerf': TinyNeRFModel,
    'nerf': NeRFModel
}

def make_model(cfg) -> nn.Module:
    return models[cfg['model']](
            embedding_dim_pos=cfg['model_hparams']['pos_emb_num'],
            embedding_dim_dir=cfg['model_hparams']['dir_emb_num'],
            hidden_dim=cfg['model_hparams']['hidden_dim'],
            pos_embedding_func=PositionalEncoding(cfg['model_hparams']['pos_emb_num']),
            dir_embedding_func=PositionalEncoding(cfg['model_hparams']['dir_emb_num'])
        )