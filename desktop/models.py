import os
import cv2
import numpy as np
import torch.cuda
from dotenv import dotenv_values, set_key
from insightface.app import FaceAnalysis
import onnxruntime
import timeit
from ellzaf_ml.models import GhostFaceNetsV2
from insightface.utils.face_align import norm_crop
from torchvision import transforms as tfs
import torch.nn.functional as F
from insightface.data import get_image

np.int = int

__alll__ = ["cosine_sim", "cosine_dist"]


class FaceSet:
    def __init__(self, folder_path, model_w, device):
        self.device = device
        self.folder_path = folder_path
        self.root_dir = os.path.curdir
        self.embeds = os.path.join(self.root_dir, "embeddgins")
        if not os.path.exists(self.embeds):
            os.makedirs(self.embeds)
        self.model_r = RecognitionModel(model_w,
                                        device=self.device)
        self.model_d = DetectionModel()

    def generate_embeddings(self):

        for person_folder in os.listdir(self.folder_path):
            person_path = os.path.join(self.folder_path, person_folder)
            embedding_person = os.path.join(self.embeds, person_folder)
            if not os.path.exists(embedding_person):
                os.makedirs(embedding_person)
            if os.path.isdir(person_path):
                for person_photo in os.listdir(person_path):
                    photo_path = os.path.join(person_path, person_photo)
                    if os.path.isfile(photo_path):
                        add_person(person_base=embedding_person,
                                   person_path=person_path,
                                   img=person_photo,
                                   detection_model=self.model_d,
                                   recognition_model=self.model_r)
        return self.embeds


class RecognitionModel:
    def __init__(self, weight_path, device):
        self.device = device
        self.model = GhostFaceNetsV2(image_size=112, num_classes=None, dropout=0).to(self.device)
        self.model.load_state_dict(torch.load(weight_path))
        self.model.eval()
        self.transform = tfs.Compose([
            tfs.ToTensor(),
            tfs.Resize((112, 112)),
            tfs.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def transform_frame(self, frame, meta):
        aligned_frame = norm_crop(frame, meta[0]["kps"])
        preprocessed = self.transform(aligned_frame)
        preprocessed = preprocessed.unsqueeze(0).to(self.device)
        return preprocessed

    def embedding(self, frame, meta):
        frame = self.transform_frame(frame, meta)
        with torch.inference_mode():
            embedding = self.model(frame)
        return F.normalize(embedding)


class Finder:
    def __init__(self, embs_path, threshold, measure, device):
        self.device = device
        self.embeds = embs_path
        self.threshold = threshold
        assert measure in __alll__
        self.measure_name = measure
        self.measure = F.cosine_similarity if measure == "cosine_sim" else self.cosine_dist
        self.persons = None

    @staticmethod
    def cosine_dist(x, y):
        return 1 - F.cosine_similarity(x, y)

    def most_similar(self, target_embedding):
        if self.persons is None:
            persons = {}
            for embed_n in os.listdir(self.embeds):
                person_name = embed_n
                person_root = os.path.join(self.embeds, embed_n)
                embed_n = os.path.join(embed_n, os.listdir(person_root)[0])
                embed_p = os.path.join(self.embeds, embed_n)
                persons[person_name] = torch.load(embed_p).to(self.device)
            self.persons = persons

        if self.measure_name == "cosine_sim":
            most_similar = (-1., "unk")
            strategy = "maximize"
        else:
            strategy = "minimize"
            most_similar = (1., "unk")

        for person_name, person_emb in self.persons.items():
            similarity = self.measure(person_emb, target_embedding)
            if strategy == "maximize":
                if similarity > most_similar[0]:
                    most_similar = (similarity, person_name)
            else:
                if similarity < most_similar[0]:
                    most_similar = (similarity, person_name)
        if strategy == "minimize":
            most_similar_meets_thresh = (most_similar[0] < self.threshold)
        else:
            most_similar_meets_thresh = (most_similar[0] > self.threshold)
        if most_similar_meets_thresh:
            return most_similar[1]
        return False


class DetectionModel:
    def __init__(self):
        self.model = FaceAnalysis(providers=['CPUExecutionProvider'], name="buffalo_sc")
        self.model.prepare(ctx_id=0, det_size=(640, 640))

    def detect_frame(self, frame):
        meta = self.model.get(frame)
        rimg = self.model.draw_on(frame, meta)
        return rimg, meta


class FaceRecognition:
    def __init__(self, model_weights, embedding_root, thresh, measure="cosine_sim", device="auto"):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model = DetectionModel()
        self.model_rec = RecognitionModel(model_weights, self.device)
        self.finder = Finder(embedding_root, thresh, measure, self.device)
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def stream(self, cap):
        start = timeit.default_timer()
        person = False
        frames = 0
        ret, frame = cap.read()
        name_cords = None
        frame, meta = self.model.detect_frame(frame)
        if len(meta) > 0:
            if isinstance(meta[0], dict):
                name_cords = (int(meta[0]["bbox"][0] + 10), int(meta[0]["bbox"][1]))
                embedding = self.model_rec.embedding(frame, meta)
                person = self.finder.most_similar(embedding)
                if person:
                    cv2.putText(frame, person, name_cords, self.font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        if not person and name_cords is not None:
            cv2.putText(frame, "Unknown", name_cords, self.font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        person = False
        frames += 1
        frame_rate = int(frames / (timeit.default_timer() - start))
        cv2.putText(frame, f"FPS: {frame_rate}", (0, 40), self.font, 1, (0, 0, 255), 3, cv2.LINE_AA)
        return frame


def add_person(person_base,
               img,
               person_path,
               detection_model,
               recognition_model):
    path = os.path.join(person_path, img)
    image = cv2.imread(path)
    face, meta = detection_model.detect_frame(image)
    embedding = recognition_model.embedding(face, meta)
    end_w = os.path.join(person_base, img).split(".")[-1]
    save_path = os.path.join(person_base, img).replace(end_w, "pth")
    torch.save(embedding, save_path)
