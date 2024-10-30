import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QFileDialog, QVBoxLayout, QMessageBox
import subprocess
import os

class Wav2LipApp(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
    
        layout = QVBoxLayout()

        self.prompt_label = QLabel("Enter text (prompt):", self)
        layout.addWidget(self.prompt_label)

        self.prompt_input = QLineEdit(self)
        layout.addWidget(self.prompt_input)

        self.video_button = QPushButton("Select Video", self)
        self.video_button.clicked.connect(self.openFileDialog)
        layout.addWidget(self.video_button)

        self.video_path = None

        self.run_button = QPushButton("Run Wav2Lip", self)
        self.run_button.clicked.connect(self.run_wav2lip)
        layout.addWidget(self.run_button)

        self.setLayout(layout)
        self.setWindowTitle('Wav2Lip Interface')
        self.setGeometry(300, 300, 400, 200)

    def openFileDialog(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi)", options=options)
        if file_name:
            self.video_path = file_name

    def run_wav2lip(self):
        if not self.video_path or not self.prompt_input.text():
            QMessageBox.warning(self, "Warning", "Please enter a prompt and select a video file.")
            return

        prompt = self.prompt_input.text()
        video_path = self.video_path

        wav2lip_command = [
            "python", "inference.py", "--checkpoint_path", "checkpoints/wav2lip_gan.pth",
            "--face", video_path, "--audio", "input_audio.wav"
        ]


        try:
            subprocess.run(wav2lip_command)
            QMessageBox.information(self, "Success", "Wav2Lip process completed.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Wav2LipApp()
    ex.show()
    sys.exit(app.exec_())
