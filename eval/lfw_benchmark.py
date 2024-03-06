from utils.data import EvalDataset
import torch
import torch.nn.functional as F
from id_rate import IdRate
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torch.nn.functional import cosine_similarity


class Evaluate:
    def __init__(self, data_path, pairs_path, root_dir, size=3):
        """
        data_path: {str} path to the image folder.
        pairs_path: {str} path to the pairs list.
        size: {int} number related to set size:
                        {
                        0: 25% dataset size
                        1: 50% dataset size
                        2: 75% dataset size
                        3: 100% dataset size
                        None: equals to 3 (100% size)
                        }
        """
        self.root_dir = root_dir
        self.data_path = data_path
        self.pairs_path = pairs_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        available_sizes = [0, 1, 2, 3]
        assert size not in available_sizes
        self.size = size
        self.dataset = self.get_dataset(self.data_path,
                                        self.pairs_path,
                                        self.size)
        self.dataloader = self.get_dataloader(self.dataset)

    def get_dataset(self, data_path, pairs_path, size):
        pairs = pd.read_csv(pairs_path, sep=' ', names=['First_image', 'Second_image', 'Issame'])
        data = EvalDataset(data_path, pairs, self.root_dir)
        return data

    def get_dataloader(self, data, batch_size=32):
        dataloader = DataLoader(data, batch_size=batch_size, pin_memory=True, num_workers=0, drop_last=True)
        return dataloader

    def compute_threshold(self, model, fpr=0.25):
        id_rate = IdRate()
        metric, threshold = id_rate.id_rate(model, fpr)
        return threshold

    def accuracy(self, model, size, metrics, fpr) -> tuple[dict, float]:

        """
        threshold: {float} computed with TPR@FPR=0.05 metric.
        size: {int} dissipate in quarters.
        metric: {str} any of {"accuracy", "f1-score", "precision", "recall"}
        fpr: {float} acceptable error miss verification rate.
        """
        threshold = 1 - self.compute_threshold(model, fpr)
        batch_res = {'tp': 0,
                     'tn': 0,
                     'fp': 0,
                     'fn': 0}
        for img1, img2, issame in self.dataloader:
            with torch.inference_mode():
                model.eval()
                img1, img2 = img1.to(self.device), img2.to(self.device)
                embd1 = F.normalize(model(img1))
                embd2 = F.normalize(model(img2))
                cos_sim = cosine_similarity(embd1, embd2, dim=1)
                labels = torch.tensor([1 if cos_sim[idx] >= threshold else -1 for idx in range(32)])
                tp = sum([1 if issame[idx] == labels[idx] == 1 else 0 for idx in range(32)])
                tn = sum([1 if issame[idx] == labels[idx] == -1 else 0 for idx in range(32)])
                fn = sum([1 if issame[idx] == 1 and labels[idx] == -1 else 0 for idx in range(32)])
                fp = sum([1 if issame[idx] == -1 and labels[idx] == 1 else 0 for idx in range(32)])
                batch_res['tp'] += tp
                batch_res['fp'] += fp
                batch_res['tn'] += tn
                batch_res['fn'] += fn
        results = {}
        for metric in metrics:
            if metric not in results.keys():
                results[metric] = 0
            if metric == 'accuracy':
                results['accuracy'] = batch_res['tp'] / len(self.dataloader)
            if metric == 'precision':
                if (batch_res['tp'] + batch_res['fp']) == 0:
                    results['precision'] = 0
                else:
                    results['precision'] = batch_res['tp'] / (batch_res['tp'] + batch_res['fp'])
            if metric == 'recall':
                if (batch_res['tp'] + batch_res['fn']) == 0:
                    results['recall'] = 0
                else:
                    results['recall'] = batch_res['tp'] / (batch_res['tp'] + batch_res['fn'])
            if metric == 'f1_score':
                precision = batch_res['tp'] / (batch_res['tp'] + batch_res['fp'])
                recall = batch_res['tp'] / (batch_res['tp'] + batch_res['fn'])
                if precision + recall == 0:
                    results['f1_score'] = 0
                else:
                    results['f1_score'] = (2 * precision * recall) / (precision + recall)
        return results, threshold
