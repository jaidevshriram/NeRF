import wandb
import torch
import numpy as np
import io
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib import cm

from torchvision.transforms.functional import resize  
from pytorch_lightning.utilities.rank_zero import rank_zero_only


def init_only_once(cfg):
    if rank_zero_only.rank != 0:
        return cfg
        
    wandb.init(config=cfg, name=cfg.experiment_name, project=cfg.project_name)
    cfg = wandb.config

    if cfg.experiment_name == "":
        cfg.update({'experiment_name': wandb.run.name}, allow_val_change=True)
    return cfg


class SegOutLogger():

    def __init__(self, logger, log_frequency=4, max_count=200) -> None:
        self.logger = logger
        self.data = []
        self.log_frequency = log_frequency
        self.max_count = max_count

    def log_image(self, X, pred, gt, batch_idx):
        pred = pred.cpu().detach().numpy()   # (B,C,H,W) -> (B,H,W)
        gt = gt.cpu().detach().numpy()

        X = resize(X.cpu().detach(), pred[0].shape) # Resize input to shape of pred
        X = X.numpy().transpose(0,2,3,1)   # (B,C,H,W) -> (B,H,W,C)
        for idx in range(0, X.shape[0], self.log_frequency):
            self.data.append((batch_idx + idx, X[idx], pred[idx], gt[idx]))  # (idx, rgb, pred, gt)

    def flush(self, stage):
        img_table = wandb.Table(columns=['ID', 'Image'])
        for item in self.data:
            mask_img = wandb.Image(item[1], masks = {
                "prediction" : {
                    "mask_data" : item[2],
                    # "class_labels" : {0: 'unknown', 1:'occupied', 2:'free'}
                },
                "ground_truth": {
                    "mask_data" : item[3],
                    # "class_labels" : {0: 'unknown', 1:'occupied', 2:'free'}
                },
            })
            
            img_table.add_data(item[0], mask_img)

        self.logger.log({f"{stage} segmentation results" : img_table})
        self.data = []

def get_plot_img(depth):

  depth = (depth[:, :, 0] - np.min(depth)) / np.max(depth)
  im = Image.fromarray(np.uint8(cm.viridis(depth) * 255))
  # plt.figure()
  # a = plt.imshow(depth[:, :, 0])
  # plt.gca().axis('off')

  # img_buf = io.BytesIO()
  # plt.savefig(img_buf, format='png')

  # im = Image.open(img_buf)
  # img_buf.close()

  return im

class GenOutLogger():

    def __init__(self, logger, log_frequency=4, max_count=200) -> None:
        self.logger = logger
        self.data = []
        self.log_frequency = log_frequency
        self.max_count = max_count

    def log_image(self, pred, mask, depth, gt, batch_idx):
        pred = pred.cpu().detach().numpy()
        gt = gt.cpu().detach().numpy()
        mask = mask.cpu().detach().numpy()
        depth = list(depth.cpu().detach().numpy())

        pred = np.clip(pred, 0, 1)

        for idx in range(0, len(pred), self.log_frequency):
            depth[idx] = get_plot_img(depth[idx])
            self.data.append([batch_idx + idx, pred[idx], mask[idx], depth[idx], gt[idx]])  # (idx, pred, gt)

    def flush(self, stage):
        img_table = wandb.Table(columns=['ID', 'Pred', 'Depth', 'GT'])
        for item in self.data:            

            img_table.add_data(item[0], wandb.Image(item[1]), wandb.Image(item[3]), wandb.Image(item[4]))

        self.logger.log({f"{stage} predictions" : img_table})
        self.data = []