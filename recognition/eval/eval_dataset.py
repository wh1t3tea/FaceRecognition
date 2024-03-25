from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import os
import os.path as osp


class EvalDataset(Dataset):
    def __init__(self, data_path, pairs_list, root_dir):
        super().__init__()
        self.data_path = data_path
        self.pairs_list = pairs_list
        self.transform = tfs.Compose([
            tfs.ToTensor(),
            tfs.Resize((112, 112)),
            tfs.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.root_dir = root_dir

    def __len__(self):
        return len(self.pairs_list)

    def __getitem__(self, idx):
        img_name1, img_name2, issame = self.pairs_list.iloc[idx]
        img1 = Image.open(os.path.join(self.root_dir, img_name1))
        img2 = Image.open(os.path.join(self.root_dir, img_name2))
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        return img1, img2, int(issame)
