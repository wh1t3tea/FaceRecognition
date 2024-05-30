from torch.utils.data import Dataset
import os.path as osp
import torchvision.transforms as T
from PIL import Image


class EvalDataset(Dataset):
    """
    Dataset class for evaluation.

    Args:
        data_path (str): Path to the data directory.
        pairs_list (DataFrame): DataFrame containing pairs of image names and their labels.
        root_dir (str): Root directory of the dataset.
    """

    def __init__(self, data_path, pairs_list, root_dir):
        super().__init__()
        self.data_path = data_path
        self.pairs_list = pairs_list
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize((112, 112)),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.root_dir = root_dir

    def __len__(self):
        return len(self.pairs_list)

    def __getitem__(self, idx):
        """
        Retrieves an item (image pair and label) from the dataset.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            tuple: A tuple containing the first image, second image, and label.
        """
        img_name1, img_name2, issame = self.pairs_list.iloc[idx]
        img1 = Image.open(osp.join(self.data_path, img_name1))
        img2 = Image.open(osp.join(self.data_path, img_name2))
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        return img1, img2, int(issame)
