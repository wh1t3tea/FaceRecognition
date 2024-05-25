from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QFormLayout, QLabel, QLineEdit, QPushButton, QMessageBox, QMainWindow

from authorization import Authorization


class AuthWindow(QMainWindow):
    is_valid = pyqtSignal()

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Login Form")
        self.setGeometry(100, 100, 300, 150)

        # Create a central widget for the main window
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create a QFormLayout to arrange the widgets
        form_layout = QFormLayout()

        # Create QLabel and QLineEdit widgets for username
        api_key_label = QLabel("API key:")
        self.api_key_field = QLineEdit()

        # Create a QPushButton for login
        login_button = QPushButton("Login")
        login_button.clicked.connect(self.login)

        # Add widgets to the form layout
        form_layout.addRow(api_key_label, self.api_key_field)
        form_layout.addRow(login_button)

        # Set the layout for the central widget
        central_widget.setLayout(form_layout)

    def login(self):
        user = Authorization(self.api_key_field.text())
        is_valid = user.login()
        if is_valid:
            QMessageBox.information(self, "Login", "Login Successful")
        else:
            QMessageBox.warning(self, "Login Failed", "No payed user with such API key")
        if is_valid:
            self.is_valid.emit()
