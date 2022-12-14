import torch
import torch.nn as nn

from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.renderer import RayBundle

from .base import BaseSampler
"""

This returns all rays instead of just a few

Input: Cameras
Output: All rays from those cameras

"""
class AllRaySampler(BaseSampler):
    
    def __init__(self, 
        img_size: int,
        n_pts_per_ray: int,
        min_depth: float,
        max_depth: float,
        batch_size: int,
        device: torch.device,
        K: torch.tensor
    ) -> None:
        
        super().__init__(img_size, n_pts_per_ray, min_depth, max_depth, batch_size, device, K)
                
    def forward(self, cameras: CamerasBase) -> RayBundle:

        points_cam = self.points_cam
        xys = self.xys

        # print(cameras.R.device, points_cam.device, cameras.T.shape)

        self.batch_size = cameras.T.shape[0]

        # Set ray origin to translation component of pose
        ray_origins = (cameras.R @ points_cam).permute(0, 2, 1) + \
            cameras.T.expand((self.img_size ** 2, self.batch_size, 3)).permute(1, 0, 2)
    
        # Rotate ray directions
        ray_directions = (cameras.R @ points_cam).permute(0, 2, 1)
        ray_directions = nn.functional.normalize(ray_directions, dim=-1)
        
        depths = torch.linspace(self.min_depth, self.max_depth, self.n_pts_per_ray).to(device=self.device)
        lengths = depths.expand(self.batch_size, self.img_size ** 2, self.n_pts_per_ray).to(device=self.device)
        
        raybundle = RayBundle(
            origins=ray_origins,
            directions=ray_directions,
            lengths=lengths,
            xys=(xys - self.img_size * 0.5) / (self.img_size * 0.5)
        )

        return raybundle