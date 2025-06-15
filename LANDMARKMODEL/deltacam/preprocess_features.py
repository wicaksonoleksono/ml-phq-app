import os
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
from tqdm import tqdm
import argparse
from collections import defaultdict

# Fungsi calculate_geometric_features tetap sama, tidak perlu diubah.


def calculate_geometric_features(landmarks, img_w, img_h):
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


def _get_subject_file_map(source_dir):
    """Memindai direktori dan mengelompokkan file gambar berdasarkan subjek dan emosi."""
    subjects = defaultdict(lambda: defaultdict(list))
    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        for img_name in os.listdir(class_path):
            try:
                subject_id = img_name.split('_')[0]
                subjects[subject_id][class_name].append(os.path.join(class_path, img_name))
            except IndexError:
                print(f"Peringatan: Melewati file dengan format nama yang tidak valid: {img_name}")
    return subjects


def _process_image(frame, face_mesh):
    """Fungsi helper untuk memproses satu gambar dan mengembalikan fiturnya."""
    if frame is None:
        return None
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.multi_face_landmarks:
        features = calculate_geometric_features(
            results.multi_face_landmarks[0].landmark, frame.shape[1], frame.shape[0]
        )
        return features
    return None


def _calculate_subject_baselines(subjects, face_mesh):
    neutral_baselines = {}
    print("📊 Menghitung baseline netral untuk setiap subjek (tanpa augmentasi)...")
    for subject_id, emotions in tqdm(subjects.items()):
        neutral_features = []
        if 'neutral' in emotions:
            for img_path in emotions['neutral']:
                frame = cv2.imread(img_path)
                if frame is None:
                    continue
                original_features = _process_image(frame, face_mesh)
                if original_features is not None:
                    neutral_features.append(original_features)
        if neutral_features:
            neutral_baselines[subject_id] = np.mean(neutral_features, axis=0)
    return neutral_baselines


def run_delta_feature_extraction(source_dir, output_csv_path, output_map_path):
    """Fungsi utama untuk ekstraksi fitur tanpa augmentasi."""
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

    subjects = _get_subject_file_map(source_dir)
    neutral_baselines = _calculate_subject_baselines(subjects, mp_face_mesh)

    print("📊 Menghitung delta fitur untuk semua emosi (tanpa augmentasi)...")
    all_delta_features = []
    class_dirs = [d for d in sorted(os.listdir(source_dir)) if os.path.isdir(os.path.join(source_dir, d))]
    class_to_idx = {name: i for i, name in enumerate(class_dirs)}

    for subject_id, emotions in tqdm(subjects.items()):
        if subject_id not in neutral_baselines:
            print(f"Peringatan: Melewati subjek '{subject_id}' karena tidak memiliki baseline netral.")
            continue
        baseline = neutral_baselines[subject_id]
        for emotion_name, img_paths in emotions.items():
            label = class_to_idx[emotion_name]
            for img_path in img_paths:
                frame = cv2.imread(img_path)
                if frame is None:
                    continue
                current_features = _process_image(frame, mp_face_mesh)
                if current_features is not None:
                    delta_features = current_features - baseline
                    all_delta_features.append(np.append(delta_features, label))
    mp_face_mesh.close()
    df = pd.DataFrame(all_delta_features)
    num_features = df.shape[1] - 1
    df.columns = [f'delta_feature_{i}' for i in range(num_features)] + ['label']
    df.to_csv(output_csv_path, index=False)
    with open(output_map_path, "w") as f:
        f.write("# File ini dibuat secara otomatis oleh preprocess_delta_features.py\n")
        f.write(f"CLASS_NAMES = {class_dirs}\n")
    print(f"\n✅ Preprocessing Delta Fitur selesai. {len(all_delta_features)} sampel diproses.")
    print(f"✅ Delta fitur berhasil disimpan ke: {output_csv_path}")
    print(f"✅ Mapping label berhasil disimpan ke: {output_map_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ekstraksi Delta Fitur Geometris dengan Kalibrasi per Subjek (Tanpa Augmentasi).")
    parser.add_argument("--input", type=str, required=True, help="Path ke direktori data gambar sumber.")
    parser.add_argument("--output_dir", type=str, default="./runs/preprocessed_data",
                        help="Direktori untuk menyimpan semua file output.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_csv_path = os.path.join(args.output_dir, "delta_emotion_features.csv")
    output_map_path = os.path.join(args.output_dir, "map_label.py")

    run_delta_feature_extraction(args.input, output_csv_path, output_map_path)
