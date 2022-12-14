import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import torchmetrics

import matplotlib.pyplot as plt

from typing import Tuple
from yacs.config import CfgNode as CN

from tqdm.auto import tqdm
from kornia.metrics import psnr

from raysampler import MonteCarloRaysampler, AllRaySampler
from raymarcher import RayMarcher

from pytorch3d.renderer import (
    FoVPerspectiveCameras, 
    RayBundle,
    ray_bundle_to_ray_points,
)

from .transformers import PositionalEncoding
from utils.activations import ShiftedSoftplus

# https://github.com/facebookresearch/pytorch3d/tree/main/projects/nerf
# https://github.com/facebookresearch/pytorch3d/blob/main/projects/nerf/nerf/raysampler.py

class TinyNeRFModel(nn.Module):
    def __init__(self, embedding_dim_pos: int, embedding_dim_dir: int, hidden_dim=256, pos_embedding_func=None, dir_embedding_func=None):
        super().__init__()

        self.input_dim_pos = 3 + 2 * 3 * embedding_dim_pos
        self.input_dim_dir = 3 + 2 * 3 * embedding_dim_dir

        self.activation = nn.ReLU

        self.block1 = nn.Sequential(
            nn.Linear(self.input_dim_pos, hidden_dim),
            self.activation(),
            nn.Linear(hidden_dim, hidden_dim),
            self.activation(),
        )

        self.alpha_mlp = nn.Sequential(nn.Linear(hidden_dim, 1), self.activation())
        # self.alpha_mlp[0].bias.data[:] = 0.0 # Try to avoid transparent predictions

        self.feat_mlp = nn.Linear(hidden_dim, hidden_dim)

        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim + self.input_dim_dir, hidden_dim),
            self.activation(),
            nn.Linear(hidden_dim, hidden_dim),
            self.activation(),
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid(),
        )

        self.pos_embedding_func = pos_embedding_func
        self.dir_embedding_func = dir_embedding_func
        self.max_batchs = 20

    def forward(self, ray_bundle: RayBundle):

        points_3d = ray_bundle_to_ray_points(ray_bundle) # B x num_rays x num_points x 3

        pos = self.pos_embedding_func(points_3d) # B x num_rays x num_points x (3 + 2 * 6 * num_functions)

        out = self.block1(pos)
        alpha = self.alpha_mlp(out)
        feat = self.feat_mlp(out) # B x num_rays x num_points x 256

        dir = self.dir_embedding_func(ray_bundle.directions) # B x num_rays x (3 + 2 * 3 * num_funcs)

        dir_expanded = dir[:, :, None, :] # B x num_rays x 1 x (3 + 2 * 3 * num_funcs)
        dir_expanded = dir_expanded.expand(*dir_expanded.shape[:2], feat.shape[2], dir_expanded.shape[-1]) # # B x num_rays x num_points x (3 + 2 * 3 * num_funcs)

        concat_inp = torch.cat((feat, dir_expanded), -1)
        rgb = self.block2(concat_inp) # B x num_rays x num_points x 3

        return alpha, rgb

    def forward_batched(self, ray_bundle: RayBundle):

        origins = ray_bundle.origins[0]
        directions = ray_bundle.directions[0]
        lengths = ray_bundle.lengths[0]
        xys = ray_bundle.xys[0]

        num_points = ray_bundle.lengths.shape[-1]
        num_rays = ray_bundle.lengths.shape[1]

        batch_idxs = torch.chunk(torch.arange(num_rays), self.max_batchs)

        batches = [RayBundle(
            origins=origins[batch_idx].unsqueeze(0),
            directions=directions[batch_idx].unsqueeze(0),
            lengths=lengths[batch_idx].unsqueeze(0),
            xys=xys[batch_idx].unsqueeze(0)
        ) for batch_idx in batch_idxs]

        outputs = [self.forward(batch) for batch in batches]

        densities = [a[0] for a in outputs]
        colours = [a[1] for a in outputs]

        densities = torch.cat(densities, dim=1)
        colours = torch.cat(colours, dim=1)

        return densities, colours
