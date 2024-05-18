import numpy as np
import cv2
from torch import nn
import torch
from insightface.app import FaceAnalysis
from face_align import norm_crop


def align_face(image, bbox: list | np.ndarray, landmarks: list | np.ndarray,
               image_size: tuple = (160, 160)) -> np.ndarray | None:
    """
    Aligns a face within the bounding box using facial landmarks.

    Args:
        image (numpy.ndarray): Input image.
        bbox (list or numpy.ndarray): Bounding box coordinates [x1, y1, x2, y2].
        landmarks (list or numpy.ndarray): Facial landmarks coordinates, e.g., [left_eye, right_eye].
        image_size (tuple, optional): Size of the aligned image. Defaults to (160, 160).

    Returns:
        numpy.ndarray or None: Aligned face image or None if bounding box coordinates are invalid.
    """
    left_eye = landmarks[0]
    right_eye = landmarks[1]

    # Calculate angle between the eyes
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.arctan2(dy, dx) * 180.0 / np.pi

    # Calculate center of eyes
    eyes_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)

    # Rotate image around the center of the eyes
    rot_matrix = cv2.getRotationMatrix2D(eyes_center, angle, scale=1)
    aligned_face = cv2.warpAffine(image, rot_matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

    # Check if any bbox coordinate is negative
    if any(coord < 0 for coord in bbox):
        return None

    x1, y1, x2, y2 = bbox

    # Crop the aligned face using bbox
    cropped_aligned_face = aligned_face[y1:y2, x1:x2]

    # Resize the cropped face to image_size
    cropped_aligned_face = cv2.resize(cropped_aligned_face, dsize=image_size)

    return torch.tensor(cropped_aligned_face)


def np_angle(a, b, c):
    """
        Calculate the angle between three points.

        Args:
            a (list or tuple): Coordinates of point a.
            b (list or tuple): Coordinates of point b.
            c (list or tuple): Coordinates of point c.

        Returns:
            float: Angle between the lines formed by the points.
    """
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)


def pred_face_pose(bbox_, landmarks_, prob_):
    """
        Predict the pose of the face based on landmarks.

        Args:
            bbox_ (list): List of bounding boxes.
            landmarks_ (list): List of landmarks for each face.
            prob_ (list): List of probabilities for each face.

        Returns:
            dict: Dictionary containing predicted angles and labels for face poses.
    """
    angle_r_list = []
    angle_l_list = []
    pred_label_list = []

    for bbox, landmarks, prob in zip(bbox_, landmarks_, prob_):
        if bbox is not None:
            ang_r = np_angle(landmarks[0], landmarks[1], landmarks[2])
            ang_l = np_angle(landmarks[1], landmarks[0], landmarks[2])
            angle_r_list.append(ang_r)
            angle_l_list.append(ang_l)
            if (int(ang_r) in range(30, 66)) and (int(ang_l) in range(30, 66)):
                pred_label = 'Frontal'
                pred_label_list.append(pred_label)
            else:
                if ang_r < ang_l:
                    pred_label = 'Left Profile'
                else:
                    pred_label = 'Right Profile'
                pred_label_list.append(pred_label)
        else:
            pred_label_list.append(None)
            angle_r_list.append(None)
            angle_l_list.append(None)

    face_d = {'angle_right': angle_r_list,
              'angle_left': angle_l_list,
              'label': pred_label_list}

    return face_d


class MTCNNFaceExtractor(nn.Module):
    """
        FaceExtractor class extracts faces from images using a given face detector model.

        Args:
            detector (nn.Module): Face detection model.
            device (str): Device to run the model on (default is 'cpu').
            img_size (tuple): Target size for aligned faces (default is (160, 160)).
    """

    def __init__(self, detector, device='cpu', img_size=(160, 160)):
        super(MTCNNFaceExtractor, self).__init__()
        self.model = detector
        self.model.device = device
        self.img_size = img_size

    def get_bbox_landmarks(self, img_batch):
        """
            Get bounding boxes and landmarks for faces in the input image batch.

            Args:
                img_batch (torch.Tensor): Batch of input images.

            Returns:
                dict: Dictionary containing lists of bounding boxes, landmarks, and probabilities.
        """
        bbox_batch, proba_batch, landmarks_batch = self.model.detect(img_batch, landmarks=True)
        results = {'bbox': [],
                   'landmarks': [],
                   'proba': []}
        for bbox, proba, landmarks in zip(bbox_batch, proba_batch, landmarks_batch):
            if bbox is not None and proba is not None and landmarks is not None:
                if proba is not None:
                    proba = proba[0]
                bbox = [int(coord) for coord in bbox[0]]
                landmarks = landmarks[0]
                results['bbox'].append(bbox)
                results['landmarks'].append(landmarks)
                results['proba'].append(proba)
            else:
                results['bbox'].append(None)
                results['landmarks'].append(None)
                results['proba'].append(None)
        return results

    def forward(self, img_batch):
        outps = self.get_bbox_landmarks(img_batch)
        aligned_faces = []
        bbox, landmarks, proba = outps['bbox'], outps['landmarks'], outps['proba']
        face_d = zip(bbox, landmarks, proba)
        angles = pred_face_pose(bbox, landmarks, proba)
        angles = angles['label']
        for idx, (face_data, angle) in enumerate(zip(face_d, angles)):
            if face_data is not None:
                if angle == 'Frontal':
                    bbox, landmarks, proba = face_data
                    aligned_faces.append(align_face(np.array(img_batch[idx]), bbox, landmarks, self.img_size))
                else:
                    aligned_faces.append(None)
            else:
                aligned_faces.append(None)
        return np.array(aligned_faces, dtype='object')


class RetinaFaceExtractor:
    def __init__(self,
                 image_size,
                 det_size=(640, 640)):
        self.detector = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.detector.prepare(ctx_id=0, det_size=det_size)
        self.image_size = image_size

    def align(self, image, to_rgb):
        if isinstance(image, str):
            img = cv2.imread(image)
        if isinstance(image, torch.Tensor):
            img = image.to('cpu').detach()
            img = img.numpy()
            img = cv2.cvtColor(img,
                               cv2.COLOR_RGB2BGR)
        else:
            img = cv2.cvtColor(np.array(image),
                               cv2.COLOR_RGB2BGR)
        if to_rgb:
            img = img[:, :, ::-1]
        faces_d = self.detector.get(img)
        highest_prob_face = faces_d[0]
        landmarks = highest_prob_face['kps']
        aligned_img = norm_crop(img,
                                landmarks,
                                image_size=self.image_size)
        return aligned_img
