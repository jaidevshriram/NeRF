import copy

import torch
import torch.nn as nn

from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.renderer import RayBundle

from .base import BaseSampler

"""

This implements the Monte Carlo Raysampler.

Input: Cameras
Output: Random set of rays from those cameras

"""
class MonteCarloRaysampler(BaseSampler):
    
    def __init__(self, 
        img_size: int,
        n_rays_per_image: int,
        n_pts_per_ray: int,
        min_depth: float,
        max_depth: float,
        batch_size: int,
        device: torch.device,
        K: torch.tensor,
        stratified: bool=False,
    ) -> None:
        
        super().__init__(img_size, n_pts_per_ray, min_depth, max_depth, batch_size, device, K)

        self.n_rays_per_image = n_rays_per_image
        self.stratified = stratified

    def forward(self, cameras: CamerasBase) -> RayBundle:
        
        rnd_idxs = torch.randperm(self.img_size ** 2, device=self.device)[:self.n_rays_per_image]

        points_cam = self.points_cam[..., rnd_idxs] # B, 3, N_rays
        points = self.xys[:, rnd_idxs, :] # B, N_rays, 2

        # Set ray origin to translation component of pose
        ray_origins = (cameras.R @ points_cam).permute(0, 2, 1) + \
            cameras.T.expand((self.n_rays_per_image, self.batch_size, 3)).permute(1, 0, 2)
        
        # Rotate ray directions
        ray_directions = (cameras.R @ points_cam).permute(0, 2, 1)
        ray_directions = nn.functional.normalize(ray_directions, dim=-1)

        depths = torch.linspace(self.min_depth, self.max_depth, self.n_pts_per_ray, device=self.device)
        lengths = depths.expand(self.batch_size, self.n_rays_per_image, self.n_pts_per_ray)
        
        if self.stratified:
            interval_size = (self.max_depth - self.min_depth) / self.n_pts_per_ray
            noise = torch.rand_like(lengths) * interval_size
            lengths = lengths + noise

        raybundle = RayBundle(
            origins=ray_origins,
            directions=ray_directions,
            lengths=lengths,
            xys=(points - self.img_size * 0.5) / (self.img_size * 0.5)
        )
        
        return raybundle