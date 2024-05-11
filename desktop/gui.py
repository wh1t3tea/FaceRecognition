import os.path
import sys

import torch.cuda
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QComboBox, QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import cv2
import numpy as np
from dotenv import load_dotenv, find_dotenv, dotenv_values, set_key
from insightface.app import FaceAnalysis
import onnxruntime
from utils import FaceRecognition, FaceSet

load_dotenv(find_dotenv())


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, camera_index):
        super().__init__()
        self.camera_index = camera_index

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        while True:
            frame = window.recognition_thread.stream(cap)
            self.change_pixmap_signal.emit(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()


class MainWindow(QWidget):
    def __init__(self, face_cfg):
        super().__init__()
        self.recognition_thread = None
        self.setWindowTitle("IdentityX")
        self.setGeometry(100, 100, 800, 600)

        self.faces_path = os.getenv("face_root")

        self.layout = QVBoxLayout()

        self.device_label = QLabel("Select Device:")
        self.device_combo = QComboBox()

        # Populate device combo with available devices

        self.camera_label = QLabel("Select Camera:")
        self.camera_combo = QComboBox()

        self.populate_device_combo()
        self.populate_camera_combo()
        if os.getenv("face_root") != "":
            self.create_database_button = QPushButton("Change your employees database")
        else:
            self.create_database_button = QPushButton("Create your employees database")
        self.create_database_button.setGeometry(150, 80, 200, 50)

        self.start_button = QPushButton("Start")

        self.video_label = QLabel(self)
        self.video_label.setFixedSize(640, 480)

        self.layout.addWidget(self.device_label)
        self.layout.addWidget(self.device_combo)
        self.layout.addWidget(self.camera_label)
        self.layout.addWidget(self.camera_combo)
        self.layout.addWidget(self.start_button)
        self.layout.addWidget(self.video_label)
        self.layout.addWidget(self.create_database_button)
        self.use_device = self.device_combo.currentIndex()
        self.embeddings = None

        self.setLayout(self.layout)

        self.video_thread = None
        self.face_cfg = face_cfg

        self.create_database_button.clicked.connect(self.on_create_database_clicked)
        self.start_button.clicked.connect(self.on_start_recognition_clicked)

    def populate_camera_combo(self):
        # Get the number of cameras available
        num_cameras = 1  # Adjust the number of cameras as per your system
        for i in range(num_cameras):
            self.camera_combo.addItem(f"Camera {i}")

    def populate_device_combo(self):
        # Add CPU
        self.device_combo.addItem("CPU")
        for i in range(torch.cuda.device_count()):
            self.device_combo.addItem(f"CUDA:{i}")

    def on_create_database_clicked(self):
        folder_path = self.select_folder()
        if folder_path:
            self.faces_path = folder_path

    def on_start_recognition_clicked(self):
        if not self.faces_path:
            self.show_warning_dialog()
        else:
            try:
                with open(".env", "w") as f:
                    f.write(f"face_root={self.faces_path}")
                    f.write(f"\nmodel_rec={model_rec}")
                self.embeddings = self.generate_embeddings()
                self.face_cfg.insert(1, self.embeddings)
                self.recognition_thread = FaceRecognition(*self.face_cfg, device=self.use_device)
                selected_camera = self.camera_combo.currentIndex()
                self.start_video_thread(selected_camera)
            except Exception as e:
                self.show_error_dialog(str(e))

    def select_folder(self):
        faces_path = QFileDialog.getExistingDirectory(self, "Select folder containing your faces", "/")
        if os.path.exists(faces_path):
            return faces_path
        return None

    def generate_embeddings(self):
        if not os.path.exists(self.faces_path):
            raise FileNotFoundError("Please select a folder containing your faces dataset.")
        faces = FaceSet(self.faces_path, self.face_cfg[0], device=self.use_device)
        return faces.generate_embeddings()

    def start_video_thread(self, selected_camera):
        if self.video_thread is None or not self.video_thread.isRunning():
            self.video_thread = VideoThread(selected_camera)
            self.video_thread.change_pixmap_signal.connect(self.update_image)
            self.video_thread.start()

    @staticmethod
    def show_warning_dialog():
        msg_box = QMessageBox()
        msg_box.setWindowTitle("Warning")
        msg_box.setText("Please select a folder with faces dataset.")
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.exec_()

    @staticmethod
    def write_to_env_file(key, value):
        with open(".env", "a") as f:
            f.write(f"\n{key}={value}")

    @staticmethod
    def show_error_dialog(error_message):
        msg_box = QMessageBox()
        msg_box.setWindowTitle("Error")
        msg_box.setText(error_message)
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.exec_()

    def update_image(self, frame):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(frame)
        self.video_label.setPixmap(qt_img)

    @staticmethod
    def convert_cv_qt(frame):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_qt_format.scaled(640, 480, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


if __name__ == '__main__':
    app = QApplication([])
    model_rec = "weights/ghostface4.pth"
    thresh = 0.3
    cfg = [model_rec, thresh]
    window = MainWindow(cfg)
    window.show()
    app.exec_()
