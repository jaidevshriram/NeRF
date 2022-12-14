import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import torchmetrics

import numpy as np

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

from utils import logging
from models import make_model

class NeRF(pl.LightningModule):
    def __init__(self, learning_rate: int, img_size: Tuple[int], cfg: dict, train_intrinsics: torch.tensor, eval_intrinsics: torch.tensor):
        super().__init__()

        # self.batch_size = cfg['raysampler']['num_rays'] * cfg['raysampler']['num_pts']

        self.model = make_model(cfg)
        self.config = cfg

        # self.log_idxs = [1, 28, 34]
        
        self.raysampler_train = MonteCarloRaysampler(
                img_size=img_size[0],
                n_rays_per_image=cfg['raysampler']['num_rays'],
                n_pts_per_ray=cfg['raysampler']['num_pts'],
                min_depth=cfg['raysampler']['min_depth'],
                max_depth=cfg['raysampler']['max_depth'],
                device=torch.device('cuda'),
                K=train_intrinsics,
                batch_size=cfg['batch_size'],
                # stratified=cfg['raysampler']['stratified']
                stratified=True
            ).to(self.device)

        self.raysampler_test = AllRaySampler(
                # img_size=img_size[0],
                img_size=800,
                n_pts_per_ray=cfg['raysampler']['num_pts'],
                min_depth=cfg['raysampler']['min_depth'],
                max_depth=cfg['raysampler']['max_depth'],
                batch_size=1,
                device=torch.device('cuda'),
                K=eval_intrinsics
            ).to(self.device)

        self.raymarcher = RayMarcher(
                near=cfg['raysampler']['min_depth'],
                far=cfg['raysampler']['max_depth'],
                include_depth=True
            ).to(self.device)

        self.learning_rate = learning_rate
        # self.save_hyperparameters()

    def setup(self, stage= None) -> None:
        if (stage == 'fit') or (stage == 'test') or (stage is None):
            self.eval_metrics = torchmetrics.MetricCollection([torchmetrics.PeakSignalNoiseRatio()])

            ## TODO: Change hard-coded image log frequency to read from config
            self.eval_img_logger = logging.get_genout_logger(self.logger.experiment)

    def render(self, raysampler, volumetric_function, raymarcher, cameras):

        # Sample rays
        rays = raysampler(cameras)

        # Run the MLP on each point to get density + RGB
        ray_density, ray_rgb = volumetric_function(ray_bundle=rays)

        # Do the integration to get the values at these points
        rgb, opacity, depth = raymarcher(densities=ray_density, rgb=ray_rgb)

        return {
            'images': rgb,
            'depths': depth,
            'densities': opacity,
            'rays': rays
        }

    def training_step(self, batch, batch_idx):

        # Generate the cameras for each view in our batch
        batch_cameras = FoVPerspectiveCameras(
                R = batch['R'], 
                T = batch['t'], 
                K = batch['intrinsics'],
                device = self.device,
            )

        # Render the RGB values for rays from the cameras 
        # For train, we have a random subset of rays
        model_output = self.render(self.raysampler_train, 
                          self.model, 
                          self.raymarcher, 
                          batch_cameras)

        rendered_images = model_output['images']
        sampled_rays = model_output['rays']
        opacity = model_output['densities']

        # opacity_images = rendered_images[..., -1] # Last chanel of image is opacity
        # rendered_images = rendered_images[..., :3] # RGB

        #VERIFY
        ray_xy_pos = sampled_rays.xys # Get XY position on the image of the rays shot out - (-1, 1)
                
        grid = ray_xy_pos.reshape(ray_xy_pos.shape[0], 1, ray_xy_pos.shape[1], 2) # (B, num_rays, 2) -> (B, 1, num_rays, 2)

        assert torch.min(grid) >= -1 and torch.max(grid) <= 1

        # Sample points from the 2D positions listed above
        rgb_points = torch.nn.functional.grid_sample(
            input=batch['RGB'],
            grid=grid,
            mode='bilinear',
            align_corners=True
        ).permute(0, 2, 3, 1).squeeze(1)

        colour_mse = nn.MSELoss()(rendered_images, rgb_points)
        loss = colour_mse

        psnr_train = psnr(rendered_images, rgb_points, max_val=1.0)

        self.log("train/train_loss", loss, on_epoch=True)
        self.log("train/PSNR", psnr_train, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "test")

    def _shared_eval(self, batch, batch_idx, prefix):

        # Generate the cameras for each view in our batch - OpenGL convention
        batch_cameras = FoVPerspectiveCameras(
                R = batch['R'], 
                T = batch['t'], 
                K = batch['intrinsics'],
                device = self.device,
            )

        # Render the RGB values for rays from the cameras 
        # For train, we have a random subset of rays
        model_output = self.render(self.raysampler_test, 
                          self.model.forward_batched, 
                          self.raymarcher, 
                          batch_cameras)

        rendered_images = model_output['images']
        sampled_rays = model_output['rays']
        weights = model_output['densities']
        depth = model_output['depths']

        num_cameras = sampled_rays.xys.shape[0]
        img_size = int(sampled_rays.xys.shape[1] ** 0.5)

        rendered_images = rendered_images.view(num_cameras, img_size, img_size, 3)
        weights = weights.view(num_cameras, img_size, img_size, 1)
        depth = depth.view(num_cameras, img_size, img_size, 1)

        ray_xy_pos = sampled_rays.xys # Get XY position on the image of the rays shot out - (-1, 1)
        grid = ray_xy_pos.view(num_cameras, img_size, img_size, 2) # B x H x W x 2

        assert torch.min(grid) >= -1 and torch.max(grid) <= 1

        # Sample points from the 2D positions listed above
        rgb_points = torch.nn.functional.grid_sample(
            input=batch['RGB'],
            grid=grid,
            mode='bilinear',
            align_corners=True
        ).permute(0, 2, 3, 1) # B x H x W x 3

        colour_mse = nn.MSELoss()(rendered_images, rgb_points)

        self.log(f"{prefix}_loss", colour_mse, prog_bar=True, on_epoch=True)

        self.eval_metrics.update(rendered_images, rgb_points)
        psnr_val = psnr(rendered_images, rgb_points, max_val=1.0)

        # if batch_idx in self.log_idxs:

        pred_img = rendered_images[0].detach().cpu().numpy()
        pred_depth = depth[0].squeeze(-1).detach().cpu().numpy()

        np.save(f"progress/{self.current_epoch}-{batch_idx}-rgb.npy", pred_img)
        np.save(f"progress/{self.current_epoch}-{batch_idx}-depth.npy", pred_depth)

        fig, ax = plt.subplots(1, 4, figsize=(10, 5))
        ax[0].imshow(rgb_points[0].detach().cpu().numpy())
        ax[0].axis('off')
        ax[0].title.set_text('Sampled points')

        ax[1].imshow(pred_img)
        ax[1].axis('off')
        ax[1].title.set_text('Rendered image')

        # ax[2].imshow(weights[0].squeeze(-1).detach().cpu().numpy())
        # ax[2].axis('off')
        # ax[2].title.set_text('Opacity Image')

        ax[2].imshow(pred_depth)
        ax[2].axis('off')
        ax[2].title.set_text('Depth Map')

        ax[3].imshow(batch['RGB'][0].permute(1, 2, 0).detach().cpu().numpy())
        ax[3].axis('off')
        ax[3].title.set_text('Actual RGB')

        plt.title(f"PSNR {psnr_val}")
        plt.show()

        self.eval_img_logger.log_image(pred=rendered_images, mask=weights, gt=rgb_points, depth=depth, batch_idx=batch_idx)
       
    def on_validation_epoch_end(self) -> None:
        self.log_metrics_and_outputs(stage='val')

    def on_test_epoch_end(self) -> None:
        self.log_metrics_and_outputs(stage='test')

    def log_metrics_and_outputs(self, stage):
        self.eval_img_logger.flush(stage)

        psnr = list(self.eval_metrics.compute().values())
        self.log(f'{stage}/psnr', psnr[0], sync_dist=True)
        
        self.eval_metrics.reset()

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        # Generate the cameras for each view in our batch - OpenGL convention
        batch_cameras = FoVPerspectiveCameras(
                R = batch['R'], 
                T = batch['t'], 
                K = batch['intrinsics'],
                device = self.device,
            )

        # Render the RGB values for rays from the cameras 
        # For train, we have a random subset of rays
        model_output = self.render(self.raysampler_test, 
                          self.model.forward_batched, 
                          self.raymarcher, 
                          batch_cameras)

        rendered_images = model_output['images']
        sampled_rays = model_output['rays']
        weights = model_output['densities']
        depth = model_output['depths']

        num_cameras = sampled_rays.xys.shape[0]
        img_size = int(sampled_rays.xys.shape[1] ** 0.5)

        rendered_images = rendered_images.view(num_cameras, img_size, img_size, 3)
        img = rendered_images[0].detach().cpu().numpy()

        plt.imshow(img)
        plt.title(f"{batch_idx}")
        plt.axis('off')
        plt.show()
        
        return img

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        shd = torch.optim.lr_scheduler.StepLR(opt, step_size=self.config['scheduler_step_size'], gamma=0.75)

        return {'optimizer':opt, 'lr_scheduler':shd, 'monitor': 'val_loss'}
        # return {'optimizer':opt, 'monitor': 'val_loss'}

if __name__ == "__main__":
    pass
    # model = Indolayout(learning_rate=1e-4)

    # input_rgb = torch.rand((4, 3, 512, 512))
    # bev = F.softmax(torch.rand((4, 3, 128, 128)), dim=1)

    # loss = model.training_step((input_rgb, None, None, bev, None), 0)
    # print(loss)