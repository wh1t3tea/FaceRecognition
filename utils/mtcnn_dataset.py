from facenet_pytorch import MTCNN
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
import cv2
from PIL import Image
import numpy as np
import os
from torchvision import transforms as tfs


class MTCNN_Dataset(Dataset):
    def __init__(self, imgs_path: str,
                 labels_path: str,
                 mode: str,
                 subset: None | str = None,
                 transform: None | torchsion.transforms = None):
        super().__init__()
        self.imgs_folder = imgs_path
        self.labels = pd.read_csv(labels_path, sep=" ", header=None)
        self.transform = transform
        self.mode = mode

        if mode == 'rec':
            if subset == 'train':
                labels = self.labels[:162770]
            elif subset == 'val':
                labels = self.labels[162770:182637]
            elif subset == 'test':
                labels = self.labels[182637:]
            elif subset == 'all':
                labels = self.labels

    def __len__(self):
        if self.mode == 'rec':
            return len(labels)
        else:
            return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.mode == 'rec':
            img_name, label = labels.iloc[idx]
            img_path = os.path.join(imgs_path, img_name)
            img = Image.open(img_path)
            if self.transform:
                img = self.transform(img)
            return img, label
        elif self.mode == 'detect':
            img_name = self.labels.iloc[:, 0].iloc[idx]
            img_path = os.path.join(imgs_path, img_name)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torch.tensor(img)
            return img, img_name


def get_dataloader(self,
                   imgs_path: str,
                   labels_path: str,
                   mode: str,
                   num_workers: int = None,
                   batch_size: int = 32) -> dict:
    if mode == 'detect':
        transforms = tfs.Compose(
            tfs.ToTensor()
        )
        train_set = MTCNN_Dataset(imgs_path,
                                  labels_path,
                                  mode=mode,
                                  subset='train',
                                  transform=transforms)
        val_set = MTCNN_Dataset(imgs_path,
                                labels_path,
                                mode=mode,
                                subset='val',
                                transform=transforms)
        test_set = MTCNN_Dataset(imgs_path,
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
