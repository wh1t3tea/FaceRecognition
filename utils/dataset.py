from facenet_pytorch import MTCNN
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
import cv2
from PIL import Image
import numpy as np
import os
from torchvision import transforms as tfs
import torchvision
import pandas as pd


class CelebaDataset(Dataset):
    def __init__(self, imgs_path: str, labels_path: str, mode: str, subset: None | str = None,
                 transform: None | torchvision.transforms = None):
        """
        Initialize CelebA dataset. You are able to use original CelebA-Face dataset:
        https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
        Or use already preprocessed version of this dataset special for facenet
        (All images were aligned. Dataset contains only frontal faces images):
        https://www.kaggle.com/datasets/wannad1e/cropped-celeba

        Args:
            imgs_path (str): Path to the folder containing images.
            labels_path (str): Path to the file containing labels.
            mode (str): 'rec' for recognition mode, 'detect' for detection mode.
            subset (str, optional): 'train', 'val', 'test', or 'all'. Defaults to None.
            transform (torchvision.transforms, optional): Transformation to apply to images. Defaults to None.
        """
        super().__init__()
        self.imgs_folder = imgs_path
        self.labels = pd.read_csv(labels_path, sep=" ", header=None)
        self.transform = transform
        self.mode = mode

        if mode == 'rec':
            if subset == 'train':
                self.labels = self.labels[:162770]
            elif subset == 'val':
                self.labels = self.labels[162770:182637]
            elif subset == 'test':
                self.labels = self.labels[182637:]
            elif subset == 'all':
                pass  # Keep all labels
            else:
                raise ValueError("Invalid subset. Choose from 'train', 'val', 'test', or 'all'.")

    def __len__(self):
        """
        Get the number of samples in the dataset.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Get the item (image and label) at the given index.

        Args:
            idx (int): Index of the item.

        Returns:
            tuple: (image, label)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.mode == 'rec':
            img_name, label = self.labels.iloc[idx]
            img_path = os.path.join(self.imgs_folder, img_name)
            img = Image.open(img_path)
            if self.transform:
                img = self.transform(img)
            return img, label
        elif self.mode == 'detect':
            img_name = self.labels.iloc[:, 0].iloc[idx]
            img_path = os.path.join(self.imgs_folder, img_name)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torch.tensor(img)
            return img, img_name


def celeba_dataloader(self,
                   img_path: str,
                   labels_path: str,
                   mode: str,
                   num_workers: int = None,
                   batch_size: int = 32) -> dict:
    """
    Get data loaders for the specified mode.

    Args:
        img_path (str): Path to the folder containing images.
        labels_path (str): Path to the file containing labels.
        mode (str): 'detect' for detection mode.
        num_workers (int, optional): Number of subprocesses to use for data loading. Defaults to None.
        batch_size (int, optional): Number of samples in each batch. Defaults to 32.

    Returns:
        dict: Dictionary containing data loaders for train, val, and test sets.
    """
    if mode != 'recognition':
        raise ValueError("Invalid mode. get_dataloader supports only recognition mode.")

    transforms = tfs.Compose([
        tfs.ToTensor()
    ])

    train_set = CelebaDataset(img_path,
                              labels_path,
                              mode=mode,
                              subset='train',
                              transform=transforms)
    val_set = CelebaDataset(img_path,
                            labels_path,
                            mode=mode,
                            subset='val',
                            transform=transforms)
    test_set = CelebaDataset(img_path,
                             labels_path,
                             mode=mode,
                             subset='test',
                             transform=transforms)

    trainloader = DataLoader(train_set,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=True,
                             pin_memory=True)
    valloader = DataLoader(val_set,
                           batch_size=batch_size,
                           num_workers=num_workers,
                           shuffle=False,
                           pin_memory=True)
    testloader = DataLoader(test_set,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=False,
                            pin_memory=True)

    return {'train': trainloader,
            'val': valloader,
            'test': testloader}


