import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
# --- KONFIGURASI ---
# Sesuaikan path ini dengan lokasi folder dataset Anda
DATASET_DIR = './Data.facial'
OUTPUT_FILE = './runs/global_neutral_baseline.npy'


def calculate_geometric_features(landmarks, img_w, img_h):
    """Fungsi ini sama persis dengan yang ada di script Anda yang lain."""
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
    mouth_pts = coords[[61, 291, 39, 181, 0, 17]]
    v_dist = np.linalg.norm(mouth_pts[2] - mouth_pts[5]) + np.linalg.norm(mouth_pts[3] - mouth_pts[4])
    h_dist = np.linalg.norm(mouth_pts[0] - mouth_pts[1])
    features.append(v_dist / (2.0 * h_dist) if h_dist > 1e-6 else 0.0)
    left_brow_pts = coords[[70, 63, 105, 66, 107]]
    right_brow_pts = coords[[336, 296, 334, 293, 300]]

    def eyebrow_angle(brow_pts):
        p_outer, p_inner = brow_pts[0], brow_pts[-1]
        return np.degrees(np.arctan2(p_inner[1] - p_outer[1], p_inner[0] - p_outer[0]))
    features.extend([eyebrow_angle(left_brow_pts), eyebrow_angle(right_brow_pts)])

    def eyebrow_curvature(brow_pts):
        p_outer, p_center, p_inner = brow_pts[0], brow_pts[2], brow_pts[-1]
        line_vec, point_vec = p_inner - p_outer, p_center - p_outer
        line_len = np.linalg.norm(line_vec)
        if line_len < 1e-6:
            return 0.0
        projection = (np.dot(point_vec, line_vec) / (line_len**2)) * line_vec
        return np.linalg.norm(point_vec - projection) / ref_dist
    features.extend([eyebrow_curvature(left_brow_pts), eyebrow_curvature(right_brow_pts)])
    features.extend([
        np.linalg.norm(coords[105, 1] - coords[159, 1]) / ref_dist,
        np.linalg.norm(coords[334, 1] - coords[386, 1]) / ref_dist,
        np.linalg.norm(coords[107] - coords[336]) / ref_dist,
        np.linalg.norm(coords[172] - coords[397]) / ref_dist,
        np.linalg.norm(coords[234] - coords[454]) / ref_dist
    ])
    return np.array(features)


def main():
    print("🚀 Memulai proses pembuatan baseline netral global...")
    neutral_dir = os.path.join(DATASET_DIR, 'neutral')
    if not os.path.isdir(neutral_dir):
        print(f"❌ Error: Direktori '{neutral_dir}' tidak ditemukan.")
        return

    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

    all_neutral_features = []
    image_files = [f for f in os.listdir(neutral_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f"📊 Ditemukan {len(image_files)} gambar di folder 'neutral'. Memproses...")
    for img_name in tqdm(image_files):
        img_path = os.path.join(neutral_dir, img_name)
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        # Proses gambar asli
        results = mp_face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            features = calculate_geometric_features(
                results.multi_face_landmarks[0].landmark, frame.shape[1], frame.shape[0])
            if features is not None:
                all_neutral_features.append(features)

        # Proses gambar yang di-flip (augmentasi untuk baseline yang lebih robust)
        flipped_frame = cv2.flip(frame, 1)
        flipped_results = mp_face_mesh.process(cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB))
        if flipped_results.multi_face_landmarks:
            flipped_features = calculate_geometric_features(
                flipped_results.multi_face_landmarks[0].landmark, flipped_frame.shape[1], flipped_frame.shape[0])
            if flipped_features is not None:
                all_neutral_features.append(flipped_features)

    mp_face_mesh.close()

    if not all_neutral_features:
        print("❌ Tidak ada fitur netral yang berhasil diekstrak. Proses dibatalkan.")
        return

    # Hitung rata-rata dari semua fitur yang terkumpul
    global_baseline = np.mean(all_neutral_features, axis=0)

    # Simpan ke file
    np.save(OUTPUT_FILE, global_baseline)

    print("\n" + "="*50)
    print(f"✅ File '{OUTPUT_FILE}' berhasil dibuat!")
    print(f"   Total sampel fitur yang diproses: {len(all_neutral_features)}")
    print(f"   Dimensi vektor baseline: {global_baseline.shape}")
    print("="*50)


if __name__ == '__main__':
    main()
