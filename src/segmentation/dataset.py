"""
dataset.py - DermaAI Segmentation Dataset

PyTorch Dataset for loading dermoscopic images and their binary lesion masks,
with resizing and augmentation support.
"""

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    def __init__(self, csv_path, image_size=256, transform=None):
        """
        csv_path: path to a CSV with columns 'image_path', 'mask_path'
        image_size: target square size for resizing
        transform: an albumentations transform (applied to both image and mask jointly)
        """
        self.df = pd.read_csv(csv_path)
        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = cv2.imread(row['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(row['mask_path'], cv2.IMREAD_GRAYSCALE)
        # Binarize just in case (0/255 -> 0/1)
        mask = (mask > 127).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            image = cv2.resize(image, (self.image_size, self.image_size))
            mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0).float()

        return image, mask
