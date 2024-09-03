# user_interface.py

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QSlider, QFileDialog, QComboBox
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPalette, QColor
import datetime

class UserInterface(QMainWindow):
    def __init__(self, central_ai):
        super().__init__()
        self.central_ai = central_ai
        self.is_ai_enabled = True
        self.console_output = ""
        self.console_height = 300
        self.current_theme = "light"
        self.weather = "Sunny"

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('AI Interface')
        self.setGeometry(100, 100, 800, 600)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        layout = QVBoxLayout()

        self.header = QLabel(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Weather: {self.weather}")
        layout.addWidget(self.header)

        self.ai_button = QPushButton('Disable AI' if self.is_ai_enabled else 'Enable AI')
        self.ai_button.clicked.connect(self.toggle_ai)
        layout.addWidget(self.ai_button)

        self.features_button = QPushButton('AI Features')
        layout.addWidget(self.features_button)

        self.upload_button = QPushButton('Upload Training Data')
        self.upload_button.clicked.connect(self.upload_training_data)
        layout.addWidget(self.upload_button)

        self.console = QLabel(self.console_output)
        self.console.setFixedHeight(self.console_height)
        layout.addWidget(self.console)

        self.console_slider = QSlider(Qt.Horizontal)
        self.console_slider.setRange(100, 500)
        self.console_slider.setValue(self.console_height)
        self.console_slider.valueChanged.connect(self.handle_console_resize)
        layout.addWidget(self.console_slider)

        self.theme_selector = QComboBox()
        self.theme_selector.addItems(["light", "dark"])
        self.theme_selector.currentTextChanged.connect(self.handle_theme_change)
        layout.addWidget(self.theme_selector)

        main_widget.setLayout(layout)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_datetime)
        self.timer.start(1000)

        self.apply_theme()

    def toggle_ai(self):
        self.is_ai_enabled = not self.is_ai_enabled
        self.ai_button.setText('Disable AI' if self.is_ai_enabled else 'Enable AI')

    def handle_console_resize(self, value):
        self.console_height = value
        self.console.setFixedHeight(value)

    def handle_theme_change(self, theme):
        self.current_theme = theme
        self.apply_theme()

    def upload_training_data(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Upload Training Data")
        if file_name:
            print(f"Training file uploaded: {file_name}")

    def update_datetime(self):
        self.header.setText(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Weather: {self.weather}")

    def apply_theme(self):
        palette = QPalette()
        if self.current_theme == "light":
            palette.setColor(QPalette.Window, QColor(255, 255, 255))
            palette.setColor(QPalette.WindowText, QColor(0, 0, 0))
            palette.setColor(QPalette.Button, QColor(76, 175, 80))
            palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
        else:
            palette.setColor(QPalette.Window, QColor(51, 51, 51))
            palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
            palette.setColor(QPalette.Button, QColor(30, 136, 229))
            palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
        self.setPalette(palette)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = UserInterface(None)  # Pass None as central_ai for now
    ui.show()
    sys.exit(app.exec_())