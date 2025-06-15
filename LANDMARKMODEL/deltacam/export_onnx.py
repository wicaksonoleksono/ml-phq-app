# File: export_onnx.py
# Tujuan: Mengonversi model PyTorch (.pth) menjadi format ONNX.

import torch
import torch.nn as nn
import argparse
import pandas as pd


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLPClassifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),  # Layer pertama lebih lebar
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 128),  # Tambah satu hidden layer
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x): return self.layers(x)


def run_export(model_path, csv_path, output_path):
    df = pd.read_csv(csv_path)
    input_dim = len(df.columns) - 1
    num_classes = len(df['label'].unique())
    print(f"Info Model: Input Dim={input_dim}, Num Classes={num_classes}")
    # Buat instance model dengan arsitektur yang benar
    model = MLPClassifier(input_dim, num_classes)
    # Muat bobot yang sudah terlatih
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set model ke mode evaluasi
    print(f"✅ Bobot dari '{model_path}' berhasil dimuat.")
    dummy_input = torch.randn(1, input_dim)
    # Ekspor ke ONNX
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

    print(f"✅ Model berhasil diekspor ke format ONNX: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ekspor Model Klasifikasi ke ONNX.")
    parser.add_argument("--model_path", type=str, default="emotion_classifier.pth",
                        help="Path ke model PyTorch (.pth) yang sudah terlatih.")
    parser.add_argument("--csv_path", type=str, default="emotion_features.csv",
                        help="Path ke file CSV untuk mendapatkan dimensi model.")
    parser.add_argument("--output_path", type=str, default="emotion_classifier.onnx",
                        help="Nama file ONNX yang akan dihasilkan.")
    args = parser.parse_args()

    run_export(args.model_path, args.csv_path, args.output_path)
