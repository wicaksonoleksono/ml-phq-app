# File: preprocess_static.py
# Tujuan: Membuat dataset "fitur statis" dari semua gambar.

import os
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
from tqdm import tqdm
import argparse


def calculate_geometric_features(landmarks, img_w, img_h):

    if not isinstance(landmarks, np.ndarray):
        coords = np.array([(lm.x * img_w, lm.y * img_h) for lm in landmarks])
    else:
        coords = landmarks

    ref_dist = np.linalg.norm(coords[133] - coords[362])
    if ref_dist < 1e-6:
        return None

    features = []

    # 1. Fitur Mata (EAR - Eye Aspect Ratio) - 2 fitur
    def eye_aspect_ratio(eye_coords):
        v1 = np.linalg.norm(eye_coords[1] - eye_coords[5])
        v2 = np.linalg.norm(eye_coords[2] - eye_coords[4])
        h = np.linalg.norm(eye_coords[0] - eye_coords[3])
        return (v1 + v2) / (2.0 * h) if h > 1e-6 else 0.0
    features.append(eye_aspect_ratio(coords[[33, 160, 158, 133, 153, 144]]))
    features.append(eye_aspect_ratio(coords[[362, 385, 387, 263, 373, 380]]))

    # 2. Fitur Alis - 3 fitur
    features.append(np.linalg.norm(coords[105, 1] - coords[159, 1]) / ref_dist)
    features.append(np.linalg.norm(coords[334, 1] - coords[386, 1]) / ref_dist)
    features.append(np.linalg.norm(coords[107] - coords[336]) / ref_dist)

    # 3. Fitur Mulut - 4 fitur
    mouth_pts = coords[[61, 291, 0, 17, 13, 14]]
    v_dist_mouth = np.linalg.norm(mouth_pts[2] - mouth_pts[3])
    h_dist_mouth = np.linalg.norm(mouth_pts[0] - mouth_pts[1])
    features.append(v_dist_mouth / h_dist_mouth if h_dist_mouth > 1e-6 else 0.0)
    features.append(h_dist_mouth / ref_dist)
    mouth_center_y = (mouth_pts[4, 1] + mouth_pts[5, 1]) / 2.0
    features.append((mouth_center_y - mouth_pts[0, 1]) / ref_dist)
    features.append((mouth_center_y - mouth_pts[1, 1]) / ref_dist)

    # 4. Fitur Hidung - 1 fitur
    eyebrow_center_pt = (coords[107] + coords[336]) / 2.0
    nose_bridge_pt = coords[6]
    features.append(np.linalg.norm(eyebrow_center_pt - nose_bridge_pt) / ref_dist)

    # 5. Fitur Garis Rahang (Jawline) - 2 fitur
    features.append(np.linalg.norm(mouth_pts[4] - coords[152]) / ref_dist)
    features.append(np.linalg.norm(coords[172] - coords[397]) / ref_dist)

    return np.array(features)


def run_static_feature_extraction(source_dir, output_csv_path, output_map_path):
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

    all_static_features = []
    class_dirs = [d for d in sorted(os.listdir(source_dir)) if os.path.isdir(os.path.join(source_dir, d))]
    class_to_idx = {name: i for i, name in enumerate(class_dirs)}

    print("📊 Mengekstrak fitur statis dari semua gambar...")
    for emotion_name in tqdm(class_dirs, desc="Processing Emotions"):
        label = class_to_idx[emotion_name]
        class_path = os.path.join(source_dir, emotion_name)

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            frame = cv2.imread(img_path)
            if frame is None:
                continue

            # Proses gambar asli
            results = mp_face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks:
                features = calculate_geometric_features(
                    results.multi_face_landmarks[0].landmark, frame.shape[1], frame.shape[0])
                if features is not None:
                    all_static_features.append(np.append(features, label))

            # Proses gambar augmentasi (flipped)
            flipped_frame = cv2.flip(frame, 1)
            flipped_results = mp_face_mesh.process(cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB))
            if flipped_results.multi_face_landmarks:
                flipped_features = calculate_geometric_features(
                    flipped_results.multi_face_landmarks[0].landmark, flipped_frame.shape[1], flipped_frame.shape[0])
                if flipped_features is not None:
                    all_static_features.append(np.append(flipped_features, label))

    mp_face_mesh.close()

    df = pd.DataFrame(all_static_features)
    num_features = df.shape[1] - 1
    df.columns = [f'static_feature_{i}' for i in range(num_features)] + ['label']
    df.to_csv(output_csv_path, index=False)

    with open(output_map_path, "w") as f:
        f.write(f"CLASS_NAMES = {class_dirs}\n")

    print(f"\n✅ Preprocessing Fitur Statis selesai. {len(all_static_features)} sampel diproses.")
    print(f"✅ Fitur statis berhasil disimpan ke: {output_csv_path}")
    print(f"✅ Mapping label berhasil disimpan ke: {output_map_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ekstraksi Fitur Geometris Statis.")
    parser.add_argument("--input", type=str, required=True, help="Path ke direktori data gambar sumber.")
    parser.add_argument("--output_dir", type=str, default="./runs/",
                        help="Direktori untuk menyimpan file output.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_csv_path = os.path.join(args.output_dir, "static_emotion_features.csv")
    output_map_path = os.path.join(args.output_dir, "map_label.py")

    run_static_feature_extraction(args.input, output_csv_path, output_map_path)
