# File: camera_app_static_model.py
# Deskripsi: Aplikasi kamera untuk deteksi emosi yang HANYA menggunakan fitur statis.
# Baseline global dan perhitungan delta DIHILANGKAN TOTAL.

import sys
import cv2
import numpy as np
import onnxruntime
import mediapipe as mp
import joblib
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap
from runs.map_label import CLASS_NAMES

# --- KONFIGURASI (SCALER PATH MUNGKIN PERLU DISESUAIKAN, LIHAT CATATAN DI BAWAH) ---
MODEL_PATH = "./runs/emotion_model.onnx"
SCALER_PATH = "./runs/scaler.pkl"  # Anda mungkin perlu mengganti nama ini
HAAR_CASCADE_PATH = "haarcascade_frontalface_default.xml"


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def calculate_geometric_features(landmarks, img_w, img_h):
    """Menghitung fitur geometris statis dari landmark."""
    if not isinstance(landmarks, np.ndarray):
        coords = np.array([(lm.x * img_w, lm.y * img_h) for lm in landmarks])
    else:
        coords = landmarks

    ref_dist = np.linalg.norm(coords[133] - coords[362])
    if ref_dist < 1e-6:
        return None

    features = []

    def eye_aspect_ratio(eye_coords):
        v1 = np.linalg.norm(eye_coords[1] - eye_coords[5])
        v2 = np.linalg.norm(eye_coords[2] - eye_coords[4])
        h = np.linalg.norm(eye_coords[0] - eye_coords[3])
        return (v1 + v2) / (2.0 * h) if h > 1e-6 else 0.0
    features.append(eye_aspect_ratio(coords[[33, 160, 158, 133, 153, 144]]))
    features.append(eye_aspect_ratio(coords[[362, 385, 387, 263, 373, 380]]))
    features.append(np.linalg.norm(coords[105, 1] - coords[159, 1]) / ref_dist)
    features.append(np.linalg.norm(coords[334, 1] - coords[386, 1]) / ref_dist)
    features.append(np.linalg.norm(coords[107] - coords[336]) / ref_dist)

    mouth_pts = coords[[61, 291, 0, 17, 13, 14]]
    v_dist_mouth = np.linalg.norm(mouth_pts[2] - mouth_pts[3])
    h_dist_mouth = np.linalg.norm(mouth_pts[0] - mouth_pts[1])
    features.append(v_dist_mouth / h_dist_mouth if h_dist_mouth > 1e-6 else 0.0)
    features.append(h_dist_mouth / ref_dist)
    mouth_center_y = (mouth_pts[4, 1] + mouth_pts[5, 1]) / 2.0
    features.append((mouth_center_y - mouth_pts[0, 1]) / ref_dist)
    features.append((mouth_center_y - mouth_pts[1, 1]) / ref_dist)

    eyebrow_center_pt = (coords[107] + coords[336]) / 2.0
    nose_bridge_pt = coords[6]
    features.append(np.linalg.norm(eyebrow_center_pt - nose_bridge_pt) / ref_dist)

    features.append(np.linalg.norm(mouth_pts[4] - coords[152]) / ref_dist)
    features.append(np.linalg.norm(coords[172] - coords[397]) / ref_dist)

    return np.array(features).reshape(1, -1)


def process_face(face_roi, face_mesh):
    """Wrapper untuk memproses ROI wajah dan mendapatkan fitur & landmark."""
    try:
        roi_h, roi_w, _ = face_roi.shape
        results = face_mesh.process(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None, None

        landmarks = results.multi_face_landmarks[0].landmark
        coords = np.array([(lm.x * roi_w, lm.y * roi_h) for lm in landmarks])

        # Panggil fungsi kalkulasi fitur yang sudah disamakan
        features = calculate_geometric_features(landmarks, roi_w, roi_h)
        if features is None:
            return None, None

        used_indices = sorted(list(set([
            133, 362, 33, 160, 158, 153, 144, 385, 387, 263, 373, 380,
            105, 159, 334, 386, 107, 336, 61, 291, 0, 17, 13, 14, 6, 152, 172, 397
        ])))
        return features, coords[used_indices]
    except Exception as e:
        print(f"Error in process_face: {e}")
        return None, None


def draw_probability_bars(frame, probabilities, class_names):
    top_indices = np.argsort(probabilities)[-5:][::-1]
    bar_x_start, bar_y_start, bar_height, bar_max_width = 20, frame.shape[0] - 150, 20, 200
    for i, idx in enumerate(top_indices):
        label, prob = class_names[idx], probabilities[idx]
        y_pos = bar_y_start + (i * (bar_height + 10))
        bar_width = int(prob * bar_max_width)
        cv2.rectangle(frame, (bar_x_start, y_pos), (bar_x_start + bar_max_width, y_pos + bar_height), (50, 50, 50), -1)
        cv2.rectangle(frame, (bar_x_start, y_pos), (bar_x_start + bar_width, y_pos + bar_height), (100, 255, 100), -1)
        text = f"{label}: {prob:.1%}"
        cv2.putText(frame, text, (bar_x_start + 5, y_pos + bar_height - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


class CameraWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Deteksi Emosi - Model Statis")
        self.setGeometry(100, 100, 800, 600)
        self.fps = 30
        try:
            self.session = onnxruntime.InferenceSession(MODEL_PATH)
            self.input_name = self.session.get_inputs()[0].name
            self.scaler = joblib.load(SCALER_PATH)  # Memuat scaler
            self.face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5)
            print("✅ Semua model dan file berhasil dimuat.")
        except Exception as e:
            print(f"❌ Gagal memuat model/file: {e}")
            return
        self.setup_ui_and_camera()

    def setup_ui_and_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.image_label = QLabel(self)
        self.layout.addWidget(self.image_label)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1000 // self.fps)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        self.perform_prediction(frame)
        self.display_image(frame)

    def perform_prediction(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face_roi = frame[y:y+h, x:x+w]
            static_features, landmarks = process_face(face_roi, self.face_mesh)
            if landmarks is not None:
                for (lx, ly) in landmarks.astype(np.int32):
                    cv2.circle(frame, (x + lx, y + ly), 1, (0, 0, 255), -1)

            if static_features is not None:
                features_scaled = self.scaler.transform(static_features)
                model_input = {self.input_name: features_scaled.astype(np.float32)}

                outputs = self.session.run(None, model_input)[0]
                probabilities = softmax(outputs[0])
                pred_idx = np.argmax(probabilities)
                label = CLASS_NAMES[pred_idx]

                display_text = f"{label} ({probabilities[pred_idx]:.1%})"
                cv2.putText(frame, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                draw_probability_bars(frame, probabilities, CLASS_NAMES)

    def display_image(self, img):
        qformat = QImage.Format.Format_BGR888
        outImage = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], qformat)
        self.image_label.setPixmap(QPixmap.fromImage(outImage))
        self.image_label.setScaledContents(True)

    def closeEvent(self, event):
        self.timer.stop()
        self.cap.release()
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraWindow()
    window.show()
    sys.exit(app.exec())
