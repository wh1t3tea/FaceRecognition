import os
import sys

import cv2
import numpy as np
import torch.cuda
from insightface.app import FaceAnalysis
import timeit
from ellzaf_ml.models import GhostFaceNetsV2
from insightface.utils.face_align import norm_crop
from torchvision import transforms as tfs
import torch.nn.functional as F

sys.path.append("auth")

np.int = int

__alll__ = ["cosine_sim", "cosine_dist"]


class FaceSet:
    """
    Class to generate embeddings for faces in a dataset folder.
    """
    def __init__(self, folder_path, model_w, device):
        """
        Initialize the FaceSet object.

        Args:
            folder_path (str): Path to the folder containing face images.
            model_w (str): Path to the recognition model weights.
            device (str): Device to use for computations (e.g., "cuda", "cpu").
        """
        self.device = device
        self.folder_path = folder_path
        self.root_dir = os.path.curdir
        self.embeds = os.path.join(self.root_dir, "embeddings")
        if not os.path.exists(self.embeds):
            os.makedirs(self.embeds)
        self.model_r = RecognitionModel(model_w,
                                        device=self.device)
        self.model_d = DetectionModel()

    def generate_embeddings(self):
        """
        Generate embeddings for faces in the dataset folder.

        Returns:
            str: Path to the embeddings folder.
        """
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
    """
    Class for the face recognition model.
    """
    def __init__(self, weight_path, device):
        """
        Initialize the RecognitionModel object.

        Args:
            weight_path (str): Path to the model weights.
            device (str): Device to use for computations (e.g., "cuda", "cpu").
        """
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
        """
        Transform a frame for recognition.

        Args:
            frame (np.ndarray): Input frame.
            meta (list): List of metadata.

        Returns:
            torch.Tensor: Transformed frame.
        """
        aligned_frame = norm_crop(frame, meta[0]["kps"])
        preprocessed = self.transform(aligned_frame)
        preprocessed = preprocessed.unsqueeze(0).to(self.device)
        return preprocessed

    def embedding(self, frame, meta):
        """
        Extract embeddings from a frame.

        Args:
            frame (np.ndarray): Input frame.
            meta (list): List of metadata.

        Returns:
            torch.Tensor: Embeddings.
        """
        frame = self.transform_frame(frame, meta)
        with torch.inference_mode():
            embedding = self.model(frame)
        return F.normalize(embedding)


class Finder:
    """
    Class to find the most similar face embeddings in a dataset.
    """
    def __init__(self, embs_path, threshold, measure, device):
        """
        Initialize the Finder object.

        Args:
            embs_path (str): Path to the folder containing face embeddings.
            threshold (float): Similarity threshold.
            measure (str): Measure to use for similarity comparison.
            device (str): Device to use for computations (e.g., "cuda", "cpu").
        """
        self.device = device
        self.embeds = embs_path
        self.threshold = threshold
        assert measure in __alll__
        self.measure_name = measure
        self.measure = F.cosine_similarity if measure == "cosine_sim" else self.cosine_dist
        self.persons = None

    @staticmethod
    def cosine_dist(x, y):
        """
        Calculate cosine distance between two vectors.

        Args:
            x (torch.Tensor): First vector.
            y (torch.Tensor): Second vector.

        Returns:
            float: Cosine distance.
        """
        return 1 - F.cosine_similarity(x, y)

    def most_similar(self, target_embedding):
        """
        Find the most similar face embedding in the dataset.

        Args:
            target_embedding (torch.Tensor): Target embedding to compare.

        Returns:
            str: Name of the most similar person, or False if no match found.
        """
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
    """
    Class for face detection model.
    """
    def __init__(self):
        """
        Initialize the DetectionModel object.
        """
        self.model = FaceAnalysis(providers=['CPUExecutionProvider'], name="buffalo_sc")
        self.model.prepare(ctx_id=0, det_size=(640, 640))

    def detect_frame(self, frame):
        """
        Detect faces in a frame.

        Args:
            frame (np.ndarray): Input frame.

        Returns:
            Tuple[np.ndarray, list]: Detected frame and metadata.
        """
        meta = self.model.get(frame)
        rimg = self.model.draw_on(frame, meta)
        return rimg, meta


class FaceRecognition:
    """
    Class for face recognition.
    """
    def __init__(self, model_weights, embedding_root, thresh, measure="cosine_sim", device="auto"):
        """
        Initialize the FaceRecognition object.

        Args:
            model_weights (str): Path to the model weights.
            embedding_root (str): Path to the folder containing face embeddings.
            thresh (float): Similarity threshold.
            measure (str): Measure to use for similarity comparison.
            device (str): Device to use for computations (e.g., "cuda", "cpu").
        """
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model = DetectionModel()
        self.model_rec = RecognitionModel(model_weights, self.device)
        self.finder = Finder(embedding_root, thresh, measure, self.device)
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def stream(self, cap):
        """
        Stream frames and perform face recognition.

        Args:
            cap: VideoCapture object.

        Returns:
            np.ndarray: Processed frame.
        """
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
        cv2.putText(frame, f"FPS: {frame_rate}", (0, 40), self.font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
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
