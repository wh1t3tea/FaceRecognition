import os
import cv2
import numpy as np
import torch.cuda
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
                embed_p = os.path.join(self.embeds, embed_n)
                persons[embed_n] = torch.load(embed_p).to(self.device)
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


def add_person(person_base,
               person_name,
               img,
               detection_model,
               recognition_model):
    img = cv2.imread(img)
    face, meta = detection_model.detect_frame(img)
    embedding = recognition_model.embedding(face, meta)
    person_file = os.path.join(person_base, person_name)
    torch.save(embedding, person_file)


# add_person("D:\PycharmProjects\FaceRecAPI\desktop\person_base", "alena", r"faces\photo_2024-04-20_17-38-12.jpg")


class FaceRecognition   :
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


if __name__ == "__main__":
    model_rec = "<RECOGNITION_MODEL_WEIGHTS>"
    embedding_path = "<YOUR_EMBEDDINGS>"
    thresh = 0.3
    stream = FaceRecognition(model_rec, embedding_path, thresh)
    stream.stream()
