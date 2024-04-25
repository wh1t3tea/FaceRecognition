import torch.cuda
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QComboBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import cv2
import numpy as np
from insightface.app import FaceAnalysis
import onnxruntime
from utils import FaceRecognition


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, camera_index):
        super().__init__()
        self.camera_index = camera_index

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        while True:
            frame = window.recogntion_thread.stream(cap)
            self.change_pixmap_signal.emit(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()


class MainWindow(QWidget):
    def __init__(self, face_cfg):
        super().__init__()

        self.setWindowTitle("Face Recognition App")
        self.setGeometry(100, 100, 800, 600)

        self.layout = QVBoxLayout()

        self.device_label = QLabel("Select Device:")
        self.device_combo = QComboBox()

        # Populate device combo with available devices
        self.populate_device_combo()

        self.camera_label = QLabel("Select Camera:")
        self.camera_combo = QComboBox()

        # Populate camera combo with available cameras
        self.populate_camera_combo()

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_recognition)

        # Label to display camera stream
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(640, 480)

        self.layout.addWidget(self.device_label)
        self.layout.addWidget(self.device_combo)
        self.layout.addWidget(self.camera_label)
        self.layout.addWidget(self.camera_combo)
        self.layout.addWidget(self.start_button)
        self.layout.addWidget(self.video_label)
        self.use_device = self.device_combo.currentIndex()

        self.setLayout(self.layout)

        self.video_thread = None
        self.face_cfg = face_cfg
        self.recogntion_thread = FaceRecognition(*self.face_cfg, device=self.use_device)

    def populate_device_combo(self):
        # Add CPU
        self.device_combo.addItem("CPU")
        for i in range(torch.cuda.device_count()):
            self.device_combo.addItem(f"CUDA:{i}")

    def populate_camera_combo(self):
        # Get the number of cameras available
        num_cameras = 4  # Adjust the number of cameras as per your system
        for i in range(num_cameras):
            self.camera_combo.addItem(f"Camera {i}")

    def start_recognition(self):
        # Start the face recognition process
        selected_camera = self.camera_combo.currentIndex()

        if self.video_thread is None or not self.video_thread.isRunning():
            self.video_thread = VideoThread(selected_camera)
            self.video_thread.change_pixmap_signal.connect(self.update_image)
            self.video_thread.start()

    def update_image(self, frame):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(frame)
        self.video_label.setPixmap(qt_img)

    def convert_cv_qt(self, frame):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_qt_format.scaled(640, 480, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


if __name__ == '__main__':
    app = QApplication([])
    model_rec = "<RECOGNITION_MODEL_WEIGHTS>"
    embedding_path = "<YOUR_EMBEDDINGS>"
    thresh = 0.3
    cfg = [model_rec, embedding_path, thresh]
    window = MainWindow(cfg)
    window.show()
    app.exec_()
