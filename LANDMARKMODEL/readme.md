# RWD-CAM: Metode Deteksi Fitur Langsung

Versi aplikasi ini menggunakan pendekatan deteksi emosi secara langsung. Model dilatih pada fitur geometris mentah yang diekstrak dari setiap gambar tanpa menggunakan referensi "wajah netral".

## Alur Kerja

Proses untuk menjalankan sistem ini terdiri dari 4 langkah utama, yang harus dijalankan secara berurutan.

### Langkah 1: Pra-pemrosesan Fitur

Skrip `preprocess_features.py` akan memindai direktori dataset, mengekstrak landmark wajah dari setiap gambar, menghitung fitur-fitur geometris (rasio mata, mulut, jarak alis, dll.), dan menyimpannya ke dalam satu file `.csv`.

**Jalankan perintah berikut dari dalam direktori `rwdcam/`**:

```bash
python preprocess_features.py --input ../Data.facial --output ./runs/rwd_features.csv
```

- `--input ../Data.facial`: Menunjuk ke folder dataset yang berada satu tingkat di atas.
- `--output ./runs/rwd_features.csv`: Menyimpan hasil ekstraksi fitur ke dalam folder `runs`.

### Langkah 2: Pelatihan Model Klasifikasi

Setelah file `.csv` dibuat, skrip `train_classifier.py` akan membaca file tersebut, melatih model klasifikasi (misalnya, SVM), dan menyimpan model yang telah dilatih beserta scaler-nya dalam format `.pkl`.

**Jalankan perintah berikut**:

```bash
python train_classifier.py --features ./runs/rwd_features.csv --output_dir ./runs
```

- Perintah ini akan menghasilkan file `rwd_emotion_model.pkl` dan `rwd_scaler.pkl` di dalam folder `runs`.

### Langkah 4: Jalankan Aplikasi Kamera

Setelah model `.onnx` siap, Anda dapat menjalankan aplikasi deteksi real-time. Aplikasi ini tidak memerlukan kalibrasi dan akan langsung mencoba mendeteksi emosi berdasarkan fitur yang diekstrak.

**Jalankan perintah berikut**:

```bash
python camera.py
```

Aplikasi akan menggunakan file `rwd_emotion_model.onnx` dan `rwd_scaler.pkl` dari folder `runs` untuk melakukan prediksi.

---

### Deskripsi File

- **`preprocess_features.py`**: Ekstraktor fitur dari dataset gambar.
- **`train_classifier.py`**: Skrip untuk melatih model dari data fitur.
- **`export_onnx.py`**: Konverter model dari format `pkl` ke `onnx`.
- **`camera.py`**: Aplikasi GUI real-time dengan webcam.
- **`haarcascade_frontalface_default.xml`**: File konfigurasi untuk deteksi wajah dasar.
