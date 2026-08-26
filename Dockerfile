# Gunakan versi Python 3.10 slim yang stabil dan hemat memori
FROM python:3.10-slim

# Set zona waktu dan hindari interaksi saat apt-get
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Jakarta

# Instalasi dependensi sistem yang wajib untuk OpenCV, Paddle, dan kompilasi InsightFace
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    build-essential \
    cmake \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Buat folder aplikasi di dalam kontainer
WORKDIR /app

# Salin file requirements terlebih dahulu untuk memanfaatkan caching Docker
COPY requirements.txt .

# Instalasi PyTorch versi CPU terlebih dahulu agar ukurannya jauh lebih kecil dan tidak gagal download
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Instalasi paket Python lainnya
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh kode aplikasi ke dalam kontainer
COPY . .

# Eksekusi server Flask
CMD ["python", "app.py"]
