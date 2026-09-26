"""
dataset.py - DermaAI Classification Dataset

PyTorch Dataset for loading HAM10000 dermoscopic images with their
diagnosis labels, encoded as integers for CrossEntropyLoss.
"""

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# Fixed class order - must stay consistent across train/val/test and all experiments
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}


class ClassificationDataset(Dataset):
    def __init__(self, csv_path, image_size=224, transform=None, label_col='diagnosis', path_col='image_path'):
        """
        csv_path: path to a CSV with at least [path_col, label_col] columns
        transform: an albumentations transform
        """
        self.df = pd.read_csv(csv_path)
        self.image_size = image_size
        self.transform = transform
        self.label_col = label_col
        self.path_col = path_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = cv2.imread(row[self.path_col])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = CLASS_TO_IDX[row[self.label_col]]

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            image = cv2.resize(image, (self.image_size, self.image_size))
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        return image, label
