import sys
import cv2
import numpy as np
import onnxruntime
import time
import timm
from torchvision import transforms
from PIL import Image

from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap
import config as cfg


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def create_timm_transform(model_name: str):
    print(f"Applying timm-recommended transforms for '{model_name}' model.")
    model = timm.create_model(model_name, pretrained=False)
    data_cfg = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_cfg, is_training=False)

    # 1. Define the same grayscale mean and std used during training
    # GRAYSCALE_MEAN = (0.5,)
    # GRAYSCALE_STD = (0.5,)

    # # 2. Insert the Grayscale transform at the beginning of the pipeline
    # transform.transforms.insert(0, transforms.Grayscale(num_output_channels=3))

    # # 3. Find and replace the default ImageNet normalization with our grayscale normalization
    # for i, t in enumerate(transform.transforms):
    #     if isinstance(t, transforms.Normalize):
    #         transform.transforms[i] = transforms.Normalize(mean=GRAYSCALE_MEAN, std=GRAYSCALE_STD)
    #         print("Successfully replaced default normalization with grayscale normalization.")
    #         break

    print("\n--- Final Inference Transforms (Consistent Grayscale Norm) ---")
    print(transform)
    print("-----------------------------------------------------------\n")
    input_size = data_cfg['input_size']
    del model
    return transform, (input_size[1], input_size[2])


def create_timm_transform(model_name: str):
    model = timm.create_model(model_name, pretrained=False)
    data_cfg = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_cfg, is_training=False)
    print(transform)

    input_size = data_cfg['input_size']
    del model
    return transform, (input_size[1], input_size[2])


class CameraWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(cfg.WINDOW_TITLE)
        self.setGeometry(100, 100, cfg.WINDOW_WIDTH, cfg.WINDOW_HEIGHT)
        self.class_names = cfg.CLASS_NAMES
        self.last_detection_time = 0
        self.last_known_faces = []
        self.last_known_predictions = {}
        try:
            self.transform, self.input_size = create_timm_transform(cfg.MODEL_NAME)
            self.input_width, self.input_height = self.input_size

            self.ort_session = onnxruntime.InferenceSession(cfg.MODEL_PATH)
            self.input_name = self.ort_session.get_inputs()[0].name
            model_input_shape = self.ort_session.get_inputs()[0].shape
            if model_input_shape[-2:] != self.input_size:
                print(
                    f"⚠️ Warning: ONNX model input shape {model_input_shape[-2:]} differs from timm recommended size {self.input_size}.")

            print(f"✅ Model Emosi ONNX '{cfg.MODEL_PATH}' berhasil dimuat.")
            print(f"✅ Transforms for '{cfg.MODEL_NAME}' created with input size {self.input_size}.")

        except Exception as e:
            print(f"❌ Gagal memuat model atau membuat transformasi: {e}")
            return

        try:
            self.face_cascade = cv2.CascadeClassifier(cfg.HAAR_CASCADE_PATH)
            print(f"✅ Classifier Wajah '{cfg.HAAR_CASCADE_PATH}' berhasil dimuat.")
        except Exception as e:
            print(f"❌ Gagal memuat Haar Cascade: {e}")
            return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("❌ Error: Tidak bisa membuka kamera.")
            return

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.image_label)

        self.timer = QTimer()
        self.timer.setInterval(1000 // cfg.VIDEO_FPS)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        display_frame = frame.copy()
        current_time = time.time()

        if (current_time - self.last_detection_time) > cfg.DETECTION_INTERVAL_SECONDS:
            self.last_detection_time = current_time
            self.last_known_predictions = {}

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.last_known_faces = self.face_cascade.detectMultiScale(
                gray_frame,
                scaleFactor=cfg.HAAR_SCALE_FACTOR,
                minNeighbors=cfg.HAAR_MIN_NEIGHBORS,
                minSize=cfg.HAAR_MIN_SIZE
            )
            for (x, y, w, h) in self.last_known_faces:
                pad_w = int(w * 0.20)
                pad_h = int(h * 0.20)
                x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
                x2, y2 = min(frame.shape[1], x + w + pad_w), min(frame.shape[0], y + h + pad_h)
                face_roi = frame[y1:y2, x1:x2]
                if face_roi.size != 0:
                    pil_image = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
                    input_tensor = self.transform(pil_image)
                    input_tensor = input_tensor.unsqueeze(0).numpy()
                    outputs = self.ort_session.run(None, {self.input_name: input_tensor})
                    scores = outputs[0][0]
                    probabilities = softmax(scores)
                    predicted_index = np.argmax(probabilities)
                    confidence = probabilities[predicted_index]
                    label_text = self.class_names[predicted_index]
                    display_text = f"{label_text}: {confidence:.2%}"

                    self.last_known_predictions[(x, y, w, h)] = display_text
        for (x, y, w, h) in self.last_known_faces:
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), cfg.BOX_COLOR, cfg.FONT_THICKNESS)
            if (x, y, w, h) in self.last_known_predictions:
                display_text = self.last_known_predictions[(x, y, w, h)]
                cv2.putText(display_frame, display_text, (x, y - 10),
                            cfg.FONT, cfg.FONT_SCALE, cfg.TEXT_COLOR, cfg.FONT_THICKNESS)

        qt_image = self.convert_cv_to_qt(display_frame)
        self.image_label.setPixmap(QPixmap.fromImage(qt_image))

    def convert_cv_to_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        return qt_format.scaled(self.image_label.width(), self.image_label.height(), Qt.AspectRatioMode.KeepAspectRatio)

    def closeEvent(self, event):
        self.timer.stop()
        self.cap.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraWindow()
    if hasattr(window, 'cap') and window.cap.isOpened():
        window.show()
    sys.exit(app.exec())
