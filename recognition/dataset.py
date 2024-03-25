from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import os
import os.path as osp


def get_dataloader(root_dir,
                   batch_size,
                   num_workers=4,
                   shuffle=True):
    transforms = T.Compose([
        T.ToTensor(),
        T.Resize((112, 112)),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    dataset = ImageFolder(root_dir, transform=transforms)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers)
    return data_loader



