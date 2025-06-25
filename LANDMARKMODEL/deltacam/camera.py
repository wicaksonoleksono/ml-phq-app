
from datetime import datetime
import json
import sys
import cv2
import numpy as np
import onnxruntime
import mediapipe as mp
import joblib
import time
import os
import hashlib
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QMessageBox  # <-- MODIFIKASI
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel,
                             QVBoxLayout, QMessageBox, QDialog, QPushButton, QHBoxLayout)
from runs.map_label import CLASS_NAMES
MODEL_PATH = "./runs/emotion_model.onnx"
SCALER_PATH = "./runs/delta_scaler.pkl"
HAAR_CASCADE_PATH = "haarcascade_frontalface_default.xml"
GLOBAL_BASELINE_PATH = "./runs/global_neutral_baseline.npy"
SAVED_FACES_DIR = "./saved_faces"
SIMILARITY_THRESHOLD = 0.3
CLASSIFICATION_INTERVAL_SECONDS = 0.5
# Di bagian paling atas file
# Di bagian paling atas file
# Di bagian paling atas file


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


class ConfirmationDialog(QDialog):
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Profil Ditemukan")
        self.setMinimumSize(250, 200)
        layout = QVBoxLayout(self)
        question_label = QLabel("Apakah ini Anda?")
        question_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = question_label.font()
        font.setPointSize(12)
        question_label.setFont(font)
        layout.addWidget(question_label)
        self.face_image_label = QLabel()
        pixmap = QPixmap(image_path)
        self.face_image_label.setPixmap(pixmap.scaled(128, 128, Qt.AspectRatioMode.KeepAspectRatio))
        self.face_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.face_image_label)
        button_layout = QHBoxLayout()
        yes_button = QPushButton("Ya, Ini Saya")
        no_button = QPushButton("Bukan, Kalibrasi Ulang")
        yes_button.clicked.connect(self.accept)
        no_button.clicked.connect(self.reject)
        button_layout.addWidget(yes_button)
        button_layout.addWidget(no_button)
        layout.addLayout(button_layout)


def calculate_geometric_features(face_roi, face_mesh):
    try:
        roi_h, roi_w, _ = face_roi.shape
        results = face_mesh.process(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))

        if not results.multi_face_landmarks:
            return None, None

        coords = np.array([(lm.x * roi_w, lm.y * roi_h) for lm in results.multi_face_landmarks[0].landmark])
        ref_dist = np.linalg.norm(coords[133] - coords[362])

        if ref_dist < 1e-6:
            return None, None

        features = []

        def eye_aspect_ratio(eye_coords):
            v1, v2, h = (np.linalg.norm(eye_coords[1]-eye_coords[5]),
                         np.linalg.norm(eye_coords[2]-eye_coords[4]),
                         np.linalg.norm(eye_coords[0]-eye_coords[3]))
            return (v1+v2)/(2.0*h) if h > 1e-6 else 0.0

        features.extend([eye_aspect_ratio(coords[[33, 160, 158, 133, 153, 144]]),
                        eye_aspect_ratio(coords[[362, 385, 387, 263, 373, 380]])])

        mouth_pts = coords[[61, 291, 39, 181, 0, 17]]
        v_dist, h_dist = (np.linalg.norm(mouth_pts[2]-mouth_pts[5]) +
                          np.linalg.norm(mouth_pts[3]-mouth_pts[4]),
                          np.linalg.norm(mouth_pts[0]-mouth_pts[1]))
        features.append(v_dist/(2.0*h_dist) if h_dist > 1e-6 else 0.0)

        left_brow_pts, right_brow_pts = coords[[70, 63, 105, 66, 107]], coords[[336, 296, 334, 293, 300]]

        def eyebrow_angle(p):
            po, pi = p[0], p[-1]
            return np.degrees(np.arctan2(pi[1]-po[1], pi[0]-po[0]))

        features.extend([eyebrow_angle(left_brow_pts), eyebrow_angle(right_brow_pts)])

        def eyebrow_curvature(p):
            po, pc, pi = p[0], p[2], p[-1]
            lv, pv = pi-po, pc-po
            ll = np.linalg.norm(lv)
            if ll < 1e-6:
                return 0.0
            proj = (np.dot(pv, lv)/(ll**2))*lv
            return np.linalg.norm(pv-proj)/ref_dist

        features.extend([eyebrow_curvature(left_brow_pts), eyebrow_curvature(right_brow_pts)])

        features.extend([
            np.linalg.norm(coords[105, 1]-coords[159, 1])/ref_dist,
            np.linalg.norm(coords[334, 1]-coords[386, 1])/ref_dist,
            np.linalg.norm(coords[107]-coords[336])/ref_dist,
            np.linalg.norm(coords[172]-coords[397])/ref_dist,
            np.linalg.norm(coords[234]-coords[454])/ref_dist
        ])

        final_features = np.array(features).reshape(1, -1)

        # Pastikan used_indices mencakup semua landmark yang dibutuhkan
        used_indices = sorted(list(set([
            # Reference points
            133, 362,
            # Left eye
            33, 160, 158, 153, 144,
            # Right eye
            385, 387, 263, 373, 380,
            # Mouth
            61, 291, 39, 181, 0, 17,
            # Left eyebrow
            70, 63, 105, 66, 107,
            # Right eyebrow
            336, 296, 334, 293, 300,
            # Additional features
            159, 386, 172, 397, 234, 454
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
        self.setWindowTitle("Deteksi Emosi Personal")
        self.setGeometry(100, 100, 800, 600)
        self.fps = 30
        self.app_state = "CHECKING"
        self.check_frames = []
        self.check_seconds = 1
        self.check_frame_count = self.check_seconds * self.fps
        self.calibration_frames = []
        self.calibration_face_images = []  # Untuk menyimpan snapshot wajah
        self.personal_offset_error = None
        self.last_classification_time = 0.0
        self.last_probabilities = np.zeros(len(CLASS_NAMES))
        self.current_user_hash = None
        self.log_session_start_time = None
        self.log_filepath = None
        self.log_data_per_second = []
        self.last_log_time = 0.0
        self.total_usage_seconds_offset = 0

        try:
            self.global_baseline = np.load(GLOBAL_BASELINE_PATH)
            self.session = onnxruntime.InferenceSession(MODEL_PATH)
            self.input_name = self.session.get_inputs()[0].name
            self.scaler = joblib.load(SCALER_PATH)
            self.face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5)
            os.makedirs(SAVED_FACES_DIR, exist_ok=True)  # Pastikan folder ada
            print("✅ Semua model dan file berhasil dimuat.")
        except Exception as e:
            print(f"❌ Gagal memuat model/file: {e}")
            self.app_state = "ERROR"
            return

        self.setup_ui_and_camera()

    def generate_face_hash(self, features_vector):
        return hashlib.sha256(features_vector.tobytes()).hexdigest()

    def save_personal_profile(self, face_image, baseline_features):
        face_hash = self.generate_face_hash(baseline_features)
        profile_dir = os.path.join(SAVED_FACES_DIR, face_hash)
        os.makedirs(profile_dir, exist_ok=True)

        cv2.imwrite(os.path.join(profile_dir, "face_image.jpg"), face_image)
        np.save(os.path.join(profile_dir, "personal_baseline.npy"), baseline_features)
        print(f"✅ Profil baru disimpan di: {profile_dir}")
        return face_hash

    def find_similar_face(self, current_features):
        for face_hash in os.listdir(SAVED_FACES_DIR):
            profile_dir = os.path.join(SAVED_FACES_DIR, face_hash)
            if os.path.isdir(profile_dir):
                try:
                    saved_baseline = np.load(os.path.join(profile_dir, "personal_baseline.npy"))
                    distance = np.linalg.norm(current_features - saved_baseline)
                    print(f"Memeriksa profil {face_hash[:10]}... Jarak: {distance:.4f}")
                    if distance < SIMILARITY_THRESHOLD:
                        saved_image_path = os.path.join(profile_dir, "face_image.jpg")
                        return saved_image_path, saved_baseline, face_hash
                except Exception as e:
                    print(f"Gagal memuat profil {face_hash}: {e}")
        return None, None, None
    # Tambahkan ini sebagai method baru di dalam class CameraWindow

    def _start_logging_session(self, user_hash):
        """Mempersiapkan dan memulai sesi logging untuk user yang teridentifikasi."""
        self.current_user_hash = user_hash
        self.log_session_start_time = time.time()

        user_log_dir = os.path.join(SAVED_FACES_DIR, self.current_user_hash, "logs")
        os.makedirs(user_log_dir, exist_ok=True)

        # --- Logika untuk melanjutkan durasi ---
        log_files = sorted([f for f in os.listdir(user_log_dir) if f.endswith('.json')])
        self.total_usage_seconds_offset = 0
        if log_files:
            last_log_path = os.path.join(user_log_dir, log_files[-1])
            try:
                with open(last_log_path, 'r') as f:
                    last_log_content = json.load(f)
                    if last_log_content.get("per_second_log"):
                        last_entry = last_log_content["per_second_log"][-1]
                        self.total_usage_seconds_offset = last_entry.get("detik_penggunaan", 0)
                        print(
                            f"ℹ️ Melanjutkan durasi dari sesi sebelumnya. Offset: {self.total_usage_seconds_offset} detik.")
            except (json.JSONDecodeError, IndexError, KeyError) as e:
                print(f"⚠️ Gagal membaca log terakhir, memulai dari 0. Error: {e}")

        # --- Membuat file log baru untuk sesi ini ---
        session_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_filepath = os.path.join(user_log_dir, f"{session_timestamp}.json")

        # Inisialisasi file JSON
        initial_data = {
            "session_start_iso": datetime.now().isoformat(),
            "per_second_log": [],
            "per_minute_summary": []
        }
        with open(self.log_filepath, 'w') as f:
            json.dump(initial_data, f, indent=4)

        print(f"📝 Sesi logging dimulai untuk user {self.current_user_hash[:10]}. File log: {self.log_filepath}")
    # Tambahkan ini sebagai method baru di dalam class CameraWindow

    def _process_and_save_log(self):
        """Memproses data log per menit dan menyimpannya ke file JSON."""
        if not self.log_data_per_second or not self.log_filepath:
            return

        # 1. Hitung ringkasan per menit
        emotion_counts = {}
        for entry in self.log_data_per_second:
            emo = entry['emosi']
            emotion_counts[emo] = emotion_counts.get(emo, 0) + 1

        total_entries = len(self.log_data_per_second)
        summary = {emo: count / total_entries for emo, count in emotion_counts.items()}

        # Tentukan menit ke berapa ini
        first_second = self.log_data_per_second[0]['detik_penggunaan']
        minute_index = (first_second - 1) // 60

        minute_summary_entry = {
            "menit_ke": minute_index + 1,
            "summary": summary
        }

        # 2. Baca file, update, dan tulis kembali
        try:
            with open(self.log_filepath, 'r+') as f:
                log_content = json.load(f)
                log_content["per_second_log"].extend(self.log_data_per_second)
                log_content["per_minute_summary"].append(minute_summary_entry)

                f.seek(0)
                json.dump(log_content, f, indent=4)
                f.truncate()

            # 3. Kosongkan buffer untuk menit selanjutnya
            self.log_data_per_second.clear()
            print(f"💾 Log untuk menit ke-{minute_index + 1} berhasil disimpan.")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"❌ Gagal menyimpan log: {e}")

    def load_profile(self, baseline_features, user_hash):
        self.personal_baseline = baseline_features
        epsilon = 1e-6
        self.scaling_factors = self.global_baseline / (self.personal_baseline + epsilon)
        offset_vector = self.personal_baseline - self.global_baseline
        self.personal_offset_error = np.linalg.norm(offset_vector)
        print("✅ Profil personal berhasil dimuat.")
        self._start_logging_session(user_hash)
        self.app_state = "RUNNING"

    def start_calibration(self):
        self.calibration_frames = []
        self.calibration_face_images = []
        self.app_state = "CALIBRATING"
        print("ℹ️ Tidak ada profil cocok / pengguna menolak. Memulai kalibrasi baru...")

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
        if self.app_state == "CHECKING":
            self.perform_initial_check(frame)
        elif self.app_state == "AWAITING_INPUT":
            self.display_image(self.last_frame_before_prompt)
            return  # Jangan proses frame baru
        elif self.app_state == "CALIBRATING":
            self.perform_offset_calibration(frame)
        elif self.app_state == "RUNNING":
            self.perform_prediction(frame)
        elif self.app_state == "ERROR":
            cv2.putText(frame, "Error: Gagal memuat file penting!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        self.display_image(frame)

    def perform_initial_check(self, frame):
        cv2.putText(frame, "Mencari profil wajah...",
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 1:
            x, y, w, h = faces[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            face_roi = frame[y:y+h, x:x+w]
            features, _ = calculate_geometric_features(face_roi, self.face_mesh)
            if features is not None:
                saved_image_path, saved_baseline, face_hash = self.find_similar_face(features)
                if saved_image_path:
                    self.app_state = "AWAITING_INPUT"
                    self.last_frame_before_prompt = frame.copy()
                    dialog = ConfirmationDialog(saved_image_path, self)

                    if dialog.exec() == QDialog.DialogCode.Accepted:
                        print("✅ Profil dikonfirmasi. Memuat profil...")
                        self.load_profile(saved_baseline, face_hash)
                    else:
                        print("ℹ️ Pengguna menolak profil. Memulai kalibrasi baru.")
                        self.start_calibration()
                    return
        self.check_frames.append(1)  # Cukup gunakan sebagai counter frame
        if len(self.check_frames) >= self.check_frame_count:
            print("ℹ️ Waktu pengecekan habis, tidak ada profil cocok. Memulai kalibrasi...")
            self.start_calibration()
            return

    def perform_offset_calibration(self, frame):
        remaining_time = max(0, (self.check_frame_count - len(self.calibration_frames)) / self.fps)
        cv2.putText(frame, f"Kalibrasi Wajah Netral: {remaining_time:.1f}s",
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 1:
            x, y, w, h = faces[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face_roi = frame[y:y+h, x:x+w]
            features, landmarks = calculate_geometric_features(face_roi, self.face_mesh)
            if landmarks is not None:
                for (lx, ly) in landmarks.astype(np.int32):
                    cv2.circle(frame, (x+lx, y+ly), 1, (0, 255, 0), -1)
            if features is not None:
                self.calibration_frames.append(features)
                self.calibration_face_images.append(frame[y:y+h, x:x+w].copy())

        if len(self.calibration_frames) >= self.check_frame_count:
            if not self.calibration_frames:
                print("❌ Kalibrasi gagal, wajah tidak terdeteksi. Mencoba lagi...")
                self.start_calibration()  # Coba lagi
                return
            personal_baseline = np.mean(self.calibration_frames, axis=0)
            face_snapshot = self.calibration_face_images[len(self.calibration_face_images)//2]  # Ambil foto dari tengah

            user_hash = self.save_personal_profile(face_snapshot, personal_baseline)
            self.load_profile(personal_baseline, user_hash)

    def perform_prediction(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        current_time = time.time()

        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_roi = frame[y:y+h, x:x+w]
            current_features, landmarks = calculate_geometric_features(face_roi, self.face_mesh)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            if landmarks is not None:
                for (lx, ly) in landmarks.astype(np.int32):
                    cv2.circle(frame, (x + lx, y + ly), 1, (0, 255, 0), -1)
            if self.app_state == "RUNNING" and (current_time - self.last_log_time) >= 1.0:
                self.last_log_time = current_time
                dominant_emotion_idx = np.argmax(self.last_probabilities)
                dominant_emotion = CLASS_NAMES[dominant_emotion_idx]
                current_usage_second = int(current_time - self.log_session_start_time) + self.total_usage_seconds_offset
                log_entry = {
                    "emosi": dominant_emotion,
                    "detik_penggunaan": current_usage_second,
                    "timestamp_iso": datetime.now().isoformat()
                }
                self.log_data_per_second.append(log_entry)
                if len(self.log_data_per_second) >= 60:
                    self._process_and_save_log()
            if (current_time - self.last_classification_time) >= CLASSIFICATION_INTERVAL_SECONDS:
                self.last_classification_time = current_time
                if current_features is not None and hasattr(self, 'personal_baseline'):
                    personal_delta = current_features - self.personal_baseline
                    scaled_delta = personal_delta * self.scaling_factors
                    features_scaled = self.scaler.transform(scaled_delta)
                    model_input = {self.input_name: features_scaled.astype(np.float32)}
                    outputs = self.session.run(None, model_input)[0]
                    self.last_probabilities = softmax(outputs[0])
                else:
                    self.last_probabilities.fill(0)
        else:
            if (current_time - self.last_classification_time) >= CLASSIFICATION_INTERVAL_SECONDS:
                self.last_probabilities.fill(0)
        pred_idx = np.argmax(self.last_probabilities)
        if len(faces) > 0 and self.last_probabilities[pred_idx] > 0.1:
            label = CLASS_NAMES[pred_idx]
            display_text = f"{label} ({self.last_probabilities[pred_idx]:.1%})"
            x, y, _, _ = faces[0]
            cv2.putText(frame, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        draw_probability_bars(frame, self.last_probabilities, CLASS_NAMES)

    def display_image(self, img):
        qformat = QImage.Format.Format_BGR888
        outImage = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], qformat)
        self.image_label.setPixmap(QPixmap.fromImage(outImage))
        self.image_label.setScaledContents(True)

    def closeEvent(self, event):
        if self.app_state == "RUNNING":
            print("ℹ️ Aplikasi ditutup, menyimpan sisa data log...")
            self._process_and_save_log()

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
