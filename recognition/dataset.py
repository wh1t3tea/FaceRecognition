from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import os
import os.path as osp
import mxnet as mx
import torch
import numpy as np
import numbers
from typing import Iterable


def get_dataloader(root_dir,
                   batch_size,
                   num_workers=4,
                   shuffle=True):
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
        dataset=train_set,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True)

    return train_loader
