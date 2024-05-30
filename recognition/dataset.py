import cv2
import pandas as pd
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import os
import mxnet as mx
import torch
import numpy as np
import numbers
from typing import Iterable


def get_dataloader(root_dir,
                   batch_size,
                   num_workers=4,
                   shuffle=True):
    """
    Get PyTorch DataLoader for the given dataset in the root directory.

    Args:
        root_dir (str): Root directory of the dataset.
        batch_size (int): Batch size.
        num_workers (int): Number of worker processes for data loading.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        DataLoader: PyTorch DataLoader.
    """
    transforms = T.Compose([
        T.ToTensor(),
        T.RandomHorizontalFlip(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    dataset = ImageFolder(root_dir, transform=transforms)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers)
    return data_loader


class MXFaceDataset(Dataset):
    def __init__(self, root_dir, local_rank):
        super(MXFaceDataset, self).__init__()
        self.transform = T.Compose(
            [T.ToPILImage(),
             T.RandomHorizontalFlip(),
             T.ToTensor(),
             T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        self.root_dir = root_dir
        self.local_rank = local_rank
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.imgidx)


def get_mx_dataloader(
        root_dir,
        batch_size,
        num_workers=2,
) -> Iterable:
    """
    Get DataLoader for the dataset in MXNet RecordIO or Image Folder format.

    Args:
        root_dir (str): Root directory of the dataset.
        batch_size (int): Batch size.
        num_workers (int): Number of worker processes for data loading.

    Returns:
        DataLoader: DataLoader for the dataset.

    """
    rec = os.path.join(root_dir, 'train.rec')
    idx = os.path.join(root_dir, 'train.idx')
    train_set = None

    # Mxnet RecordIO
    if os.path.exists(rec) and os.path.exists(idx):
        train_set = MXFaceDataset(root_dir=root_dir, local_rank=0)

    # Image Folder
    else:
        transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        train_set = ImageFolder(root_dir, transform)

    train_loader = DataLoader(
        shuffle=True,
        dataset=train_set,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True)

    return train_loader


class CelebaDataset(Dataset):
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

    def __init__(self,
                 imgs_path: str,
                 labels_path: str,
                 mode: str,
                 subset: None | str = None,
                 transform: None | torchvision.transforms = None):

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


def celeba_dataloader(
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

    transforms = T.Compose([
        T.ToTensor()
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
