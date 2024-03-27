from .eval_dataset import EvalDataset
import torch
import torch.nn.functional as F
from .id_rate import IdRate
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torch.nn.functional import cosine_similarity
import os.path as osp
import os


class Evaluate:
    def __init__(self, pairs_dir, tar_far_dir, fpr):
        """
        pairs_data_path: {str} path to the image folder.
        pairs_annot_path: {str} path to the pairs list.
        """

        self.fpr = fpr

        pairs_annot_path = osp.join(pairs_dir, "lfw_pair.txt")
        pairs_data_path = osp.join(pairs_dir, "lfw")

        tar_far_annot = osp.join(tar_far_dir, "query_anno.txt")
        tar_far_query = osp.join(tar_far_dir, "celeba_aligned")
        tar_far_distractors = osp.join(tar_far_dir, "distractors")

        self.id_rate_cfg = [tar_far_annot,
                            tar_far_distractors,
                            tar_far_query]

        self.pairs_cfg = {'labels': pairs_annot_path,
                          'data': pairs_data_path,
                          'root': pairs_dir}

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.dataset = self.get_dataset()
        self.dataloader = DataLoader(self.dataset,
                                     batch_size=32,
                                     pin_memory=True,
                                     num_workers=0,
                                     drop_last=True)

    def get_dataset(self):
        pairs = pd.read_csv(self.pairs_cfg['labels'], sep=' ', names=['First_image', 'Second_image', 'Issame'])
        data = EvalDataset(self.pairs_cfg['data'], pairs, self.pairs_cfg['root'])
        return data

    def compute_threshold(self,
                          model):
        id_rate = IdRate(*self.id_rate_cfg)
        metric, threshold = id_rate.id_rate(model, self.fpr)
        return threshold

    def accuracy(self,
                 model,
                 metrics,
                 fpr) -> tuple[dict, float]:

        """
        threshold: {float} computed with TPR@FPR=0.05 metric.
        size: {int} dissipate in quarters.
        metric: {str} any of {"accuracy", "f1-score", "precision", "recall"}
        fpr: {float} acceptable error miss verification rate.
        """
        threshold = 1 - self.compute_threshold(model)
        print(threshold)
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
                results['accuracy'] = (batch_res['tp'] + batch_res['tn']) / (batch_res['tp'] + batch_res['fp'] + batch_res['fn'] + batch_res['tn'])
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
