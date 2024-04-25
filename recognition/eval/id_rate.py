import torch
from torchvision import transforms as T
from torch.nn.functional import cosine_similarity, normalize
import os
from collections import defaultdict
from PIL import Image
import torch.nn.functional as F


def preprocess_image(img_path):
    img = Image.open(img_path)
    transform = T.Compose([
        T.ToTensor(),
        T.Resize((112, 112)),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    img = transform(img)
    return img


def compute_ir(cosine_query_pos, cosine_query_neg, cosine_query_distractors, fpr=0.1):
    """
    compute identification rate using precomputed cosine similarities between pairs
    at a given fpr
    params:
      cosine_query_pos: cosine similarities between positive pairs from query
      cosine_query_neg: cosine similarities between negative pairs from query
      cosine_query_distractors: cosine similarities between negative pairs
                                from query and distractors
      fpr: false positive rate at which to compute TPR
    output:
      float: threshold for given fpr
      float: TPR at given FPR
    """
    # Combine negative pairs from query and distractors
    all_neg_cos = torch.cat([torch.tensor(cosine_query_distractors), torch.tensor(cosine_query_neg)])

    # Find the threshold efficiently without sorting the entire list
    if isinstance(fpr, list):
        tprs = {}
        thresholds = []
        for fpr_ in fpr:
            threshold_idx = int(fpr_ * len(all_neg_cos))
            threshold, _ = torch.kthvalue(all_neg_cos, threshold_idx)

            # Convert the threshold to a Python float
            threshold = threshold.item()

            # Calculate TPR at the given FPR
            true_positives = sum(1 for x in cosine_query_pos if x < threshold)
            tpr = true_positives / len(cosine_query_pos)
            tprs[fpr_] = tpr
            thresholds.append(threshold)
        return tprs, thresholds

    threshold_idx = int(fpr * len(all_neg_cos))
    threshold, _ = torch.kthvalue(all_neg_cos, threshold_idx)

    # Convert the threshold to a Python float
    threshold = threshold.item()

    # Calculate TPR at the given FPR
    true_positives = sum(1 for x in cosine_query_pos if x < threshold)
    tpr = true_positives / len(cosine_query_pos)

    return tpr, threshold


class IdRate:
    """
    This metric is related to LFW-benchmark. It allows you to choose threshold value of cosine distance to verify persons.
    You are able to choose acceptable verification error. It might be best to select FPR values between 0.05-0.2.
    For computing this metric you need to download special query-distractor LFW set.
    You can find this dataset in my kaggle works:
    "https://www.kaggle.com/datasets/wannad1e/ssssas"
    """

    def __init__(self, annot_path,
                 distractor_path,
                 query_path):
        self.annot_path = annot_path
        self.distractors_path = distractor_path
        self.query_path = query_path
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    def query_names(self):
        with open(self.annot_path, 'r') as f:
            query_lines = f.readlines()[1:]
            query_lines = [x.strip().split(',') for x in query_lines]
            query_img_names = [x[0] for x in query_lines]
            return query_img_names

    def query_dict(self):
        with open(self.annot_path, 'r') as f:
            query_lines = f.readlines()[1:]
            query_lines = [x.strip().split(',') for x in query_lines]
            query_dict = defaultdict(list)
            for img_name, img_class in query_lines:
                query_dict[img_class].append(img_name)
            return query_dict

    def distractor_names(self):
        return os.listdir(self.distractors_path)

    def compute_embeddings(self, model, images_list, root_dir, normal=True):
        '''
        compute embeddings from the trained model for list of images.
        params:
          model: trained nn model that takes images and outputs embeddings
          images_list: list of images paths to compute embeddings for
        output:
          list: list of model embeddings. Each embedding corresponds to images
                names from images_list
        '''
        embeds = []
        with torch.inference_mode():
            for img_path in images_list:
                model.eval()
                img_path_r = os.path.join(root_dir, img_path)
                img = preprocess_image(img_path_r).to(self.device)
                if normal:
                    embed = F.normalize(model(img.unsqueeze(0))).to('cpu')
                else:
                    embed = model(img.unsqueeze(0)).to('cpu')
                embeds.append((embed, img_path))
        return embeds

    def compute_cosine_query_pos(self, query_dict, query_embeddings):
        """
        compute cosine similarities between positive pairs from query (stage 1)
        params:
          query_dict: dict {class: [image_name_1, image_name_2, ...]}. Key: class in
                      the dataset. Value: images corresponding to that class
          query_img_names: list of images names
          query_embeddings: list of embeddings corresponding to query_img_names
        output:
          list of floats: similarities between embeddings corresponding
                          to the same people from query list
        """
        grouped_embeds = [[0] for x in range(len(query_dict.values()))]
        embed_names = [x[1] for x in query_embeddings]
        cos_list = []
        for idx, values in enumerate(query_dict.values()):
            for value in values:
                if value in embed_names:
                    if not isinstance(grouped_embeds[idx][0], torch.Tensor):
                        grouped_embeds[idx] = [query_embeddings[embed_names.index(value)][0]]
                    else:
                        grouped_embeds[idx].append(query_embeddings[embed_names.index(value)][0])
        grouped_embeds = [x for x in grouped_embeds if len(x) > 1]
        for group in grouped_embeds:
            embed = group[0]
            for embedding in group[1:]:
                cos_list.append(1 - cosine_similarity(embed.to(self.device), embedding.to(self.device)).to('cpu'))
        return cos_list

    def compute_cosine_query_neg(self, query_dict, query_embeddings):
        """
        compute cosine similarities between negative pairs from query (stage 2)
        params:
          query_dict: dict {class: [image_name_1, image_name_2, ...]}. Key: class in
                      the dataset. Value: images corresponding to that class
          query_img_names: list of images names
          query_embeddings: list of embeddings corresponding to query_img_names
        output:
          list of floats: similarities between embeddings corresponding
                          to different people from query list
        """
        grouped_embeds = [[0] for x in range(len(query_dict.values()))]
        embed_names = [x[1] for x in query_embeddings]
        cos_list = []
        embds = [x[0] for x in query_embeddings]
        for idx, values in enumerate(query_dict.values()):
            for value in values:
                if value in embed_names:
                    if not isinstance(grouped_embeds[idx][0],  torch.Tensor):
                        grouped_embeds[idx] = [query_embeddings[embed_names.index(value)][0]]
                    else:
                        grouped_embeds[idx].append(query_embeddings[embed_names.index(value)][0])
        grouped_embeds = [x for x in grouped_embeds if len(x) > 1]
        tensor_embeds = torch.stack(embds)
        for group in grouped_embeds:
            for embedding in group:
                mask = torch.any(tensor_embeds != embedding, dim=-1)
                embeds_ = tensor_embeds[mask]
                cos_list.append(1 - cosine_similarity(embedding.to(self.device), embeds_.to(self.device)).to('cpu'))
        return torch.cat(cos_list)

    def compute_cosine_query_distractors(self, query_embeddings, distractors_embeddings):
        """
        compute cosine similarities between negative pairs from query and distractors
        (stage 3)
        params:
          query_embeddings: list of embeddings corresponding to query_img_names
          distractors_embeddings: list of embeddings corresponding to distractors_img_names
        output:
          list of floats: similarities between pairs of people (q, d), where q is
                          embedding corresponding to photo from query, d —
                          embedding corresponding to photo from distractors
        """
        cos_list = []
        distractors_embeddings = torch.stack([x[0] for x in distractors_embeddings])
        for q_emb in query_embeddings:
            dist_embeds_mask = torch.any(distractors_embeddings != q_emb[0], dim=-1)
            cos_list.append(
                1 - cosine_similarity(q_emb[0].to(self.device),
                                      distractors_embeddings[dist_embeds_mask].to(self.device)).to('cpu'))
        return torch.cat(cos_list)

    def id_rate(self, model, fpr):
        q_embeddings = self.compute_embeddings(model,
                                               self.query_names(),
                                               self.query_path)
        d_embeddings = self.compute_embeddings(model,
                                               self.distractor_names(),
                                               self.distractors_path)
        cos_query_pos = self.compute_cosine_query_pos(self.query_dict(),
                                                      q_embeddings)
        cos_query_neg = self.compute_cosine_query_neg(self.query_dict(),
                                                      q_embeddings)
        cos_query_distractors = self.compute_cosine_query_distractors(q_embeddings, d_embeddings)
        print(compute_ir(cos_query_pos, cos_query_neg, cos_query_distractors, fpr=[0.01, 0.05, 0.1]))
        return compute_ir(cos_query_pos, cos_query_neg, cos_query_distractors, fpr=fpr)
