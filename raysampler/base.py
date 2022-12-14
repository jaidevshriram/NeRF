import torch
import torch.nn as nn

'''
This is a base class for all raysamplers - it creates a meshgrid, backprojects points, for use in custom raysamplers.
'''
class BaseSampler(nn.Module):

    def __init__(self, 
                img_size: int, 
                n_pts_per_ray: int, 
                min_depth: float, 
                max_depth: float, 
                batch_size: int, 
                device: torch.device,
                K = torch.tensor
                ):
        super().__init__()

        self.img_size = img_size
        self.n_pts_per_ray = n_pts_per_ray
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.batch_size = batch_size
        self.device = device
        self.K = K.to(self.device)

        # Create 2D mesh
        range_coord = torch.linspace(0, self.img_size-1, self.img_size, device=self.device)
        i, j = torch.meshgrid(range_coord, range_coord)
        
        i, j = i.T.flatten(), j.T.flatten()
        
        points = torch.stack([i, j], dim=-1).expand((self.batch_size, self.img_size ** 2, 2)).to(device=self.device)
        xys = torch.stack([i, j], dim=-1).expand((self.batch_size, self.img_size ** 2, 2)).to(device=self.device)

        # 2D points -> Homogenous coordinates
        points_homo = torch.cat((points, torch.ones((self.batch_size, self.img_size ** 2, 1), device=self.device)), dim=-1).float()

        # Backproject points
        points_cam = torch.linalg.inv(self.K[:, :3, :3]) @ points_homo.permute(0, 2, 1)
        points_cam[:, 1:, :] *= -1
    
        self.points_cam = points_cam.to(self.device)
        self.xys = xys.to(self.device)
    