# File: camera_app_calibration.py
# VERSI FINAL: Disesuaikan untuk menggunakan DELTA FITUR yang sama dengan preprocessing. (TANPA FLIP)

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

# --- KONFIGURASI ---
MODEL_PATH = "./runs/emotion_model.onnx"
SCALER_PATH = "./runs/delta_scaler.pkl"
HAAR_CASCADE_PATH = "haarcascade_frontalface_default.xml"
GLOBAL_BASELINE_PATH = "./runs/global_neutral_baseline.npy"


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# =============================================================================
# FUNGSI KALKULASI FITUR (WAJIB SAMA DENGAN PREPROCESSING)
# =============================================================================


def calculate_geometric_features(face_roi, face_mesh):
    """
    Fungsi ini memproses ROI wajah, mengekstrak landmarks, dan menghitung
    vektor fitur geometris yang sama persis dengan yang digunakan saat training.
    """
    try:
        roi_h, roi_w, _ = face_roi.shape
        results = face_mesh.process(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None, None

        # --- 1. Konversi landmarks ke koordinat numpy ---
        coords = np.array([(lm.x * roi_w, lm.y * roi_h) for lm in results.multi_face_landmarks[0].landmark])

        # --- 2. Hitung Fitur (LOGIKA INI DIAMBIL LANGSUNG DARI SCRIPT PREPROCESSING) ---
        ref_dist = np.linalg.norm(coords[133] - coords[362])
        if ref_dist < 1e-6:
            return None, None

        features = []

        # Eye Aspect Ratio (EAR)
        def eye_aspect_ratio(eye_coords):
            v1 = np.linalg.norm(eye_coords[1] - eye_coords[5])
            v2 = np.linalg.norm(eye_coords[2] - eye_coords[4])
            h = np.linalg.norm(eye_coords[0] - eye_coords[3])
            return (v1 + v2) / (2.0 * h) if h > 1e-6 else 0.0
        features.append(eye_aspect_ratio(coords[[33, 160, 158, 133, 153, 144]]))
        features.append(eye_aspect_ratio(coords[[362, 385, 387, 263, 373, 380]]))

        # Mouth Aspect Ratio (MAR)
        mouth_pts = coords[[61, 291, 39, 181, 0, 17]]
        v_dist = np.linalg.norm(mouth_pts[2] - mouth_pts[5]) + np.linalg.norm(mouth_pts[3] - mouth_pts[4])
        h_dist = np.linalg.norm(mouth_pts[0] - mouth_pts[1])
        features.append(v_dist / (2.0 * h_dist) if h_dist > 1e-6 else 0.0)

        # Eyebrow Angle
        left_brow_pts = coords[[70, 63, 105, 66, 107]]
        right_brow_pts = coords[[336, 296, 334, 293, 300]]

        def eyebrow_angle(brow_pts):
            p_outer, p_inner = brow_pts[0], brow_pts[-1]
            return np.degrees(np.arctan2(p_inner[1] - p_outer[1], p_inner[0] - p_outer[0]))
        features.extend([eyebrow_angle(left_brow_pts), eyebrow_angle(right_brow_pts)])

        # Eyebrow Curvature
        def eyebrow_curvature(brow_pts):
            p_outer, p_center, p_inner = brow_pts[0], brow_pts[2], brow_pts[-1]
            line_vec, point_vec = p_inner - p_outer, p_center - p_outer
            line_len = np.linalg.norm(line_vec)
            if line_len < 1e-6:
                return 0.0
            projection = (np.dot(point_vec, line_vec) / (line_len**2)) * line_vec
            return np.linalg.norm(point_vec - projection) / ref_dist
        features.extend([eyebrow_curvature(left_brow_pts), eyebrow_curvature(right_brow_pts)])

        # Other Distances
        features.extend([
            np.linalg.norm(coords[105, 1] - coords[159, 1]) / ref_dist,
            np.linalg.norm(coords[334, 1] - coords[386, 1]) / ref_dist,
            np.linalg.norm(coords[107] - coords[336]) / ref_dist,
            np.linalg.norm(coords[172] - coords[397]) / ref_dist,
            np.linalg.norm(coords[234] - coords[454]) / ref_dist
        ])

        final_features = np.array(features).reshape(1, -1)
        # --- 3. Kumpulkan Indeks Landmark untuk Visualisasi ---
        used_indices = sorted(list(set([
            133, 362, 33, 160, 158, 153, 144, 385, 387, 263, 373, 380,
            61, 291, 39, 181, 0, 17, 70, 63, 105, 66, 107, 336, 296,
            334, 293, 300, 159, 386, 172, 397, 234, 454
        ])))

        return final_features, coords[used_indices]

    except Exception as e:
        print(f"Error in calculate_geometric_features: {e}")
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
        self.setWindowTitle("Deteksi Emosi - Hybrid Baseline")
        self.setGeometry(100, 100, 800, 600)
        self.fps = 30
        self.global_baseline = None
        self.personal_offset_error = None
        self.is_calibrating = True
        self.calibration_frames = []
        self.calibration_seconds = 3
        self.calibration_frame_count = self.calibration_seconds * self.fps
        try:
            self.global_baseline = np.load(GLOBAL_BASELINE_PATH)
            print(f"✅ Baseline global berhasil dimuat dari {GLOBAL_BASELINE_PATH}.")
        except FileNotFoundError:
            print(f"❌ File '{GLOBAL_BASELINE_PATH}' tidak ditemukan! Aplikasi tidak bisa melakukan prediksi.")
            self.is_calibrating = False
        try:
            self.session = onnxruntime.InferenceSession(MODEL_PATH)
            self.input_name = self.session.get_inputs()[0].name
            self.scaler = joblib.load(SCALER_PATH)
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
        if self.global_baseline is None:
            cv2.putText(frame, "Error: GLOBAL BASELINE TIDAK DITEMUKAN!",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif self.is_calibrating:
            self.perform_offset_calibration(frame)
        else:
            self.perform_prediction(frame)
        self.display_image(frame)

    def perform_offset_calibration(self, frame):
        remaining_time = max(0, (self.calibration_frame_count - len(self.calibration_frames)) / self.fps)
        text = f"Kalibrasi Wajah Netral: {remaining_time:.1f}s"
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 1:
            x, y, w, h = faces[0]
            face_roi = frame[y:y+h, x:x+w]
            features, _ = calculate_geometric_features(face_roi, self.face_mesh)
            if features is not None:
                self.calibration_frames.append(features)
        if len(self.calibration_frames) >= self.calibration_frame_count:
            if self.calibration_frames:
                self.personal_baseline = np.mean(self.calibration_frames, axis=0)
                # --- PERUBAHAN LOGIKA UTAMA ADA DI SINI ---
                epsilon = 1e-6
                self.scaling_factors = self.global_baseline / (self.personal_baseline + epsilon)
                # Skor keunikan tetap bisa dihitung untuk informasi
                offset_vector = self.personal_baseline - self.global_baseline
                self.personal_offset_error = np.linalg.norm(offset_vector)

                print("✅ Kalibrasi Skala selesai. Faktor skala personal disimpan.")
                print(f"   Skor Keunikan Wajah (Offset Error): {self.personal_offset_error:.4f}")
            else:
                self.personal_offset_error = -1
                self.personal_baseline = None
                self.scaling_factors = None  # Pastikan faktor skala tidak ada jika gagal
                print("❌ Kalibrasi gagal, wajah tidak terdeteksi.")

            self.is_calibrating = False

    def perform_prediction(self, frame):
        if self.personal_offset_error is not None:
            offset_text = f"Skor Keunikan: {self.personal_offset_error:.2f}"
            color = (0, 255, 0) if self.personal_offset_error < 5.0 else (0, 255, 255)
            cv2.putText(frame, offset_text, (frame.shape[1] - 300, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            if self.personal_offset_error > 5.0:
                cv2.putText(frame, "Wajah Unik, Akurasi Mungkin Bervariasi",
                            (frame.shape[1] - 300, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face_roi = frame[y:y+h, x:x+w]
            current_features, landmarks = calculate_geometric_features(face_roi, self.face_mesh)
            if landmarks is not None:
                for (lx, ly) in landmarks.astype(np.int32):
                    cv2.circle(frame, (x + lx, y + ly), 1, (0, 255, 0), -1)
            if (current_features is not None and
                hasattr(self, 'personal_baseline') and self.personal_baseline is not None and
                    hasattr(self, 'scaling_factors') and self.scaling_factors is not None):
                personal_delta = current_features - self.personal_baseline
                scaled_delta = personal_delta * self.scaling_factors
                features_scaled = self.scaler.transform(scaled_delta)
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
