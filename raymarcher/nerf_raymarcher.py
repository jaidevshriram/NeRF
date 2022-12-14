import torch
import torch.nn as nn

class RayMarcher(nn.Module):
    def __init__(self, include_depth=True, near=0.1, far=6.0):
        super().__init__()
        self.eps = 1e-10
        self.depth = include_depth
        self.near = near
        self.far = far
        
    def calculate_depth(self, weights: torch.Tensor) -> torch.Tensor:
        
        num_points = weights.shape[-1]
        depth_vals = torch.linspace(self.near, self.far, num_points).to(weights.device)
        
        depth = weights * depth_vals[None, None, :]
        depth = torch.sum(depth, dim=-1)
        
        return depth

    def forward(self,
            densities: torch.Tensor, # N x num_rays x num_points x 1
            rgb: torch.Tensor, # N x num_rays x num_points x 3
        ) -> torch.Tensor:

        device = rgb.device
        densities = densities.squeeze(-1)
        
        num_images = densities.shape[0]
        num_rays = densities.shape[1]
        num_pts = densities.shape[2]
        
        # Calculate the light upto that point
        transmittance = torch.cumprod((1 + self.eps) - densities, dim=-1)
        
        # Shift the transmittance by one
        transmittance = torch.cat([torch.ones((num_images, num_rays, 1), device=device), transmittance], dim=-1)[..., :num_pts]

        # \int_t_{near}^{t} * density at the point
        weights = densities * transmittance
        
        # print(weights.shape, rgb.shape, weights[..., None].shape)
        
        # RGB(t) = \int_t_{near}^{t} * density at the point * color
        # RGB = sum of RGB(t)
        rgb_calc = torch.sum(weights[..., None] * rgb, dim=-2)
        
        opacity_calc = 1.0 - torch.prod(1.0 - densities, dim=-1, keepdim=True)
        
        if self.depth:
            depth = self.calculate_depth(weights)
            return rgb_calc, opacity_calc, depth
                
        return rgb_calc, opacity_calc
