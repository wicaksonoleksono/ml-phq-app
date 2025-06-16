# File: train_and_export.py
# Tujuan: Membaca data, melatih model, mengevaluasi, dan langsung mengekspor
#         model terbaik ke format ONNX dalam satu pipeline.

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
import argparse
import joblib
import os


class EmotionFeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self): return len(self.features)
    def __getitem__(self, idx): return self.features[idx], self.labels[idx]


class MLPClassifier(nn.Module):

    def __init__(self, input_dim, num_classes):
        super(MLPClassifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x): return self.layers(x)


def get_dataloaders(csv_path, batch_size, output_dir, test_split=0.2):

    df = pd.read_csv(csv_path)
    X = df.drop('label', axis=1).values
    y = df['label'].values
    input_dim = X.shape[1]
    num_classes = len(np.unique(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=test_split, random_state=42, stratify=y_train)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    scaler_path = os.path.join(output_dir, 'delta_scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"✅ Scaler berhasil disimpan ke '{scaler_path}'")

    train_ds = EmotionFeatureDataset(X_train, y_train)
    val_ds = EmotionFeatureDataset(X_val, y_val)
    test_ds = EmotionFeatureDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, input_dim, num_classes


def export_model_to_onnx(model, input_dim, output_path):
    """
    Mengekspor model PyTorch yang sudah dilatih ke format ONNX.
    """
    # Pastikan model berada di CPU dan dalam mode evaluasi
    model.to(torch.device('cpu'))
    model.eval()

    # Buat input dummy sesuai dengan dimensi input model
    dummy_input = torch.randn(1, input_dim)

    print("\n🚀 Mengekspor model terbaik ke ONNX...")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input_features'],
            output_names=['output_logits'],
            dynamic_axes={'input_features': {0: 'batch_size'},
                          'output_logits': {0: 'batch_size'}}
        )
        print(f"✅ Model berhasil diekspor ke: {output_path}")
    except Exception as e:
        print(f"❌ Gagal mengekspor model ke ONNX: {e}")


def train_evaluate_and_export(config):
    """
    Orkestrasi seluruh proses: training, evaluasi, dan ekspor.
    """
    # Pastikan direktori output ada
    os.makedirs(config.output_dir, exist_ok=True)

    # 1. Mempersiapkan Data
    train_loader, val_loader, test_loader, input_dim, num_classes = get_dataloaders(
        config.csv_path, config.batch_size, config.output_dir
    )

    # 2. Inisialisasi Model, Loss, dan Optimizer
    model = MLPClassifier(input_dim, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True)

    best_val_loss = float('inf')
    model_save_path = os.path.join(config.output_dir, 'best_emotion_model.pth')

    print(f"🚀 Memulai Training untuk {config.num_epochs} epoch di {device}...")

    # 3. Loop Training
    for epoch in range(config.num_epochs):
        model.train()
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss, val_preds, val_labels = 0, [], []
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"✨ Model terbaik baru ditemukan! Val Loss: {avg_val_loss:.4f}. Disimpan ke '{model_save_path}'")

        if (epoch + 1) % 10 == 0:
            acc = accuracy_score(val_labels, val_preds)
            f1 = f1_score(val_labels, val_preds, average='macro')
            print(f"Epoch [{epoch+1}/{config.num_epochs}], Val Loss: {avg_val_loss:.4f}, Val Acc: {acc:.4f}, Val F1: {f1:.4f}")
    print("\n🧪 Menjalankan Testing pada model TERBAIK...")
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    print("\n--- Laporan Klasifikasi Final (dari model terbaik) ---")
    print(classification_report(test_labels, test_preds, digits=4))

    onnx_output_path = os.path.join(config.output_dir, 'emotion_model.onnx')
    export_model_to_onnx(model, input_dim, onnx_output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training dan Ekspor Klasifikasi Emosi dari Fitur Geometris.")
    parser.add_argument("--csv_path", type=str, default="./emotion_features.csv",
                        help="Path ke file CSV berisi fitur.")
    parser.add_argument("--output_dir", type=str, default="./runs",
                        help="Direktori untuk menyimpan semua hasil (model .pth, scaler .pkl, dan .onnx).")
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=100)

    args = parser.parse_args()
    train_evaluate_and_export(args)
