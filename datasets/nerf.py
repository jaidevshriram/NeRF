import os
from typing import List, Tuple
import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader

import PIL.ImageOps
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

import matplotlib.pyplot as plt

g = torch.Generator()
g.manual_seed(42)

class NerfDataset(Dataset):
    def __init__(self, data_dir:str, files:List[str], img_size:Tuple[int], stage: str='train') -> None:
        super().__init__()
        self.data_dir = data_dir
        self.files = files
        self.img_size = img_size
        self.split = stage
        self.prefix = {
            'train': 0,
            'val': 1,
            'test': 2
        }

        color_transforms = [
            transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ]

        self.color_transforms = transforms.Compose(color_transforms)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        file_name = self.files[index]

        img_path = os.path.join(self.data_dir, "rgb", f"{file_name}.png")
        pose_path = os.path.join(self.data_dir, "pose", f"{file_name}.txt")
        intrinsics_path = os.path.join(self.data_dir, "intrinsics.txt")

        intrinsic_scale = self.img_size[0] / 800

        if self.split != 'test':
            rgb = Image.open(img_path).convert('RGB')
            rgb = self.color_transforms(rgb)
        else:
            rgb = []

        pose = np.float32(np.loadtxt(pose_path)) # OpenCV convention +z forward, -y up, +x right
        intrinsics = np.float32(np.loadtxt(intrinsics_path))
        intrinsics[0, 2] *= intrinsic_scale
        intrinsics[1, 2] *= intrinsic_scale
        intrinsics[0, 0] *= intrinsic_scale
        intrinsics[1, 1] *= intrinsic_scale

        pose[:, 1:3] *= -1 # OpenGL convention : -z forward, +y up, +x right
        # pose[:, :2] *= -1 # Pytorch3D convention - +z forward, +y up, -x right
        R = pose[:3, :3]
        t = pose[:3, -1]

        return {
            'id': index,
            'RGB': rgb,
            'R': R,
            't': t,
            'intrinsics': intrinsics
        }

class NerfDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, split_dir: str, img_size: Tuple[int], batch_size: int = 1, num_workers: int=0):
        super().__init__()
        self.data_dir = data_dir
        self.split_dir = split_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.save_hyperparameters()

    def setup(self, stage=None):
        with open(os.path.join(self.split_dir, 'train.txt'), 'r') as f:
            train_files = f.read().splitlines()
        with open(os.path.join(self.split_dir, 'val.txt'), 'r') as f:
            val_files = f.read().splitlines()
        with open(os.path.join(self.split_dir, 'test.txt'), 'r') as f:
            test_files = f.read().splitlines()

        if stage == 'fit' or stage is None:
            self.nerf_train = NerfDataset(self.data_dir, train_files, self.img_size, stage='train')
            self.nerf_val = NerfDataset(self.data_dir, val_files, (800, 800), stage='val')

        if stage == 'test' or stage == 'predict' or stage is None:
            self.nerf_test = NerfDataset(self.data_dir, test_files, self.img_size, stage='test')

    def train_dataloader(self):
        return DataLoader(self.nerf_train, batch_size=self.batch_size, \
            shuffle=True, generator=g, drop_last=True, \
            num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.nerf_val, batch_size=1, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.nerf_test, batch_size=1, num_workers=self.num_workers, pin_memory=True)

    def predict_dataloader(self):
        return DataLoader(self.nerf_test, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def teardown(self, stage=None):
        super().teardown(stage)
