# Sistem Pengawasan Kamera Ganda (Deteksi Wajah & Plat Nomor)

Proyek ini adalah sebuah aplikasi web berbasis **Flask** yang dirancang untuk melakukan pengawasan cerdas menggunakan dua aliran kamera terpisah secara bersamaan (misalnya menggunakan DroidCam/IP Camera).

Sistem ini menggabungkan dua model kecerdasan buatan terdepan:
1. **Face Engine (InsightFace)**: Mendeteksi dan mengenali wajah secara real-time.
2. **Plate Engine (YOLOv8 + PaddleOCR)**: Mendeteksi kendaraan, melokalisasi letak plat nomor (menggunakan YOLOv8 yang telah di-finetune khusus), dan membaca teks plat nomor dengan presisi tinggi (menggunakan PaddleOCR dengan dukungan algoritma *CLAHE* dan ekstraksi *Regex* khusus plat nomor Indonesia).

## 🚀 Fitur Utama
- **Dual Camera Support**: Memproses dua kamera IP secara bersamaan.
- **Deteksi Wajah Presisi Tinggi**: Ditenagai oleh model `buffalo_l` dari InsightFace.
- **Smart OCR Plat Nomor**: 
  - YOLOv8 Custom Fine-Tuned Model untuk bounding-box plat nomor.
  - CLAHE Pre-processing untuk menerangkan plat nomor yang berada di area gelap/berbayang.
  - Ekstraksi Regex pintar untuk membuang teks yang tidak relevan (seperti tanggal pajak) dan memvalidasi format plat nomor Indonesia.
- **Database SQLite**: Pencatatan data otomatis.

## 🛠️ Prasyarat (Requirements)
Sistem ini membutuhkan pustaka berikut (lihat `requirements.txt` untuk lebih lengkapnya):
- Python 3.8+
- Flask
- Ultralytics (YOLOv8)
- PaddleOCR & PaddlePaddle
- InsightFace
- OpenCV (`opencv-python`)
- ONNX Runtime

## ⚙️ Konfigurasi Lingkungan (.env)
Buat atau edit file `.env` di direktori utama proyek dengan format berikut:
```env
INVERT_CAMERA=true
PLATE_MODEL_PATH=models/best_finetuned.pt
PLATE_CAMERA_URL=http://10.10.30.68:4747/video
FACE_CAMERA_URL=http://10.10.30.72:4747/video
```
*(Ganti URL kamera sesuai dengan IP DroidCam atau IP Camera Anda).*

## 🏃 Cara Menjalankan
1. Pastikan Anda telah mengaktifkan *virtual environment*:
   ```bash
   venv\Scripts\activate
   ```
2. Jalankan server Flask utama:
   ```bash
   python app.py
   ```
3. Buka *browser* dan akses alamat lokal yang diberikan (biasanya `http://127.0.0.1:5000/` atau `http://0.0.0.0:5000/`).

## 🧪 Pengujian Offline (Inference)
Jika Anda hanya ingin mengetes ketepatan OCR pada gambar statis (tanpa kamera):
1. Masukkan gambar-gambar plat nomor (mendukung ekstensi standar dan `.HEIC`) ke dalam folder `Testing_plat/`.
2. Jalankan *script* inferensi:
   ```bash
   python inference_plat.py
   ```
3. Cek hasil ekstraksi teks dan *bounding box* di dalam folder `Testing_plat_output/`.
