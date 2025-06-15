import sys
import cv2
import numpy as np
import onnxruntime
import time
import timm
import mediapipe as mp
from torchvision import transforms
from PIL import Image

from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap
import config as cfg  # Asumsikan file config.py Anda ada


def softmax(x):
    """Menghitung softmax untuk array numpy."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def create_timm_transform(model_name: str):
    """
    Membuat pipeline transformasi dari timm dan menambahkan konversi Grayscale.
    """
    model = timm.create_model(model_name, pretrained=False)
    data_cfg = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_cfg, is_training=False)
    # transform.transforms.insert(0, transforms.Grayscale(num_output_channels=3))
    print("✅ Pipeline transformasi yang digunakan:")
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
            self.ort_session = onnxruntime.InferenceSession(cfg.MODEL_PATH)
            self.input_name = self.ort_session.get_inputs()[0].name
            output_shape = self.ort_session.get_outputs()[0].shape
            jumlah_kelas = output_shape[1]  # Asumsi bentuk output adalah [batch_size, num_classes]
            print(f"✅ Model ini memiliki {jumlah_kelas} kelas output.")
            # -----------------------------

            print(f"✅ Model Emosi ONNX '{cfg.MODEL_PATH}' berhasil dimuat.")
            print(f"✅ Model Emosi ONNX '{cfg.MODEL_PATH}' berhasil dimuat.")

        except Exception as e:
            print(f"❌ Gagal memuat model atau transformasi: {e}")
            return
        try:
            self.face_cascade = cv2.CascadeClassifier(cfg.HAAR_CASCADE_PATH)
            print(f"✅ Classifier Wajah (Haar) '{cfg.HAAR_CASCADE_PATH}' berhasil dimuat.")
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,  # False lebih baik untuk video real-time
                max_num_faces=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print("✅ MediaPipe Face Mesh model berhasil dimuat.")

        except Exception as e:
            print(f"❌ Gagal memuat detektor wajah: {e}")
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

    def align_face_roi(self, face_roi):
        try:
            roi_h, roi_w, _ = face_roi.shape
            results = self.face_mesh.process(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))

            if not results.multi_face_landmarks:
                return None
            landmarks_unnormalized = np.array(
                [(lm.x * roi_w, lm.y * roi_h) for lm in results.multi_face_landmarks[0].landmark]
            )
            left_eye = landmarks_unnormalized[133]
            right_eye = landmarks_unnormalized[362]
            dY = right_eye[1] - left_eye[1]
            dX = right_eye[0] - left_eye[0]
            angle = np.degrees(np.arctan2(dY, dX))

            eyes_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
            M = cv2.getRotationMatrix2D(eyes_center, angle, scale=1.0)
            aligned_face = cv2.warpAffine(face_roi, M, (roi_w, roi_h), flags=cv2.INTER_CUBIC)
            return aligned_face
        except Exception:
            return None

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        display_frame = frame.copy()
        current_time = time.time()
        if (current_time - self.last_detection_time) > cfg.DETECTION_INTERVAL_SECONDS:
            self.last_detection_time = current_time
            self.last_known_predictions = {}
            # Tahap 1: Deteksi cepat dengan Haar Cascade
            # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.last_known_faces = self.face_cascade.detectMultiScale(
                # gray_frame,
                frame,
                scaleFactor=cfg.HAAR_SCALE_FACTOR,
                minNeighbors=cfg.HAAR_MIN_NEIGHBORS,
                minSize=cfg.HAAR_MIN_SIZE
            )
            for (x, y, w, h) in self.last_known_faces:
                face_roi = frame[y:y+h, x:x+w]
                if face_roi.size != 0:
                    aligned_face = self.align_face_roi(face_roi)
                    if aligned_face is not None:
                        # Konversi wajah yang sudah lurus & grayscale (via transform) untuk model
                        pil_image = Image.fromarray(cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB))
                        input_tensor = self.transform(pil_image).unsqueeze(0).numpy()
                        # Jalankan inferensi ONNX
                        outputs = self.ort_session.run(None, {self.input_name: input_tensor})
                        scores = outputs[0][0]
                        probabilities = softmax(scores)
                        predicted_index = np.argmax(probabilities)
                        confidence = probabilities[predicted_index]
                        label_text = self.class_names[predicted_index]

                        display_text = f"{label_text}: {confidence:.2%}"

                        self.last_known_predictions[(x, y, w, h)] = display_text

        # Gambar kotak dan teks di frame display
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
        self.face_mesh.close()  # Penting: Tutup model MediaPipe
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraWindow()
    if hasattr(window, 'cap') and window.cap.isOpened():
        window.show()
    sys.exit(app.exec())
