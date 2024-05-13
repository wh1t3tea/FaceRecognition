import os.path
import sys
from PyQt5.QtGui import QIcon
import torch.cuda
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QComboBox, QFileDialog, QMessageBox, QHBoxLayout, QGridLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import cv2
import numpy as np
from dotenv import load_dotenv, find_dotenv, dotenv_values, set_key
from insightface.app import FaceAnalysis
import onnxruntime
from models import FaceRecognition, FaceSet


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
        icon_path = "static/app_icon.ico"
        self.setWindowIcon(QIcon(icon_path))

        self.faces_path = os.getenv("face_root")

        layout = QGridLayout(self)
        layout.setSpacing(20)

        header_label = QLabel("IdentityX - Face Recognition System", self)
        header_label.setAlignment(Qt.AlignCenter)
        header_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #333;")
        layout.addWidget(header_label, 0, 0, 1, 2)

        device_label = QLabel("Select Device:", self)
        device_label.setStyleSheet("font-size: 16px; color: #666;")
        layout.addWidget(device_label, 1, 0)
        self.device_combo = QComboBox(self)
        self.populate_device_combo()
        layout.addWidget(self.device_combo, 1, 1)

        camera_label = QLabel("Select Camera:", self)
        camera_label.setStyleSheet("font-size: 16px; color: #666;")
        layout.addWidget(camera_label, 2, 0)
        self.camera_combo = QComboBox(self)
        self.populate_camera_combo()
        layout.addWidget(self.camera_combo, 2, 1)

        self.create_database_button = QPushButton("Create/Change Database", self)

        self.create_database_button.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                color: #fff;
                background-color: #007bff;
                border: none;
                padding: 10px 20px;
                min-width: 100px;
                max-width: 200px;
                margin-left: 260 px;
                margin-top: 20px;
            }
        """)
        self.create_database_button.clicked.connect(self.on_create_database_clicked)
        layout.addWidget(self.create_database_button, 3, 0, 1, 2)

        self.start_button = QPushButton("Start Recognition", self)
        self.start_button.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                color: #fff;
                background-color: #007bff;
                border: none;
                padding: 10px 20px;
                min-width: 100px;
                max-width: 200px;
                margin-left: 260 px;
                margin-top: 5px;
            }
        """)
        self.start_button.clicked.connect(self.on_start_recognition_clicked)
        layout.addWidget(self.start_button, 4, 0, 1, 2)

        self.video_label = QLabel(self)
        self.video_label.setScaledContents(True)
        self.video_label.setMinimumSize(640, 480)
        layout.addWidget(self.video_label, 5, 0, 1, 2)
        self.video_thread = None

        self.setLayout(layout)
        self.setMinimumSize(800, 800)
        self.setMaximumSize(800, 800)

        self.use_device = self.device_combo.currentIndex()
        self.embeddings = None
        self.face_cfg = face_cfg
    def populate_camera_combo(self):
        num_cameras = 1
        for i in range(num_cameras):
            self.camera_combo.addItem(f"Camera {i}")

    def populate_device_combo(self):
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
        qt_img = self.convert_cv_qt(frame)
        self.video_label.setPixmap(qt_img)

    @staticmethod
    def convert_cv_qt(frame):
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
