import os
import cv2
import numpy as np
import re
import logging
from ultralytics import YOLO
from paddleocr import PaddleOCR

# Try importing pillow_heif for HEIC support
try:
    from pillow_heif import register_heif_opener
    from PIL import Image
    register_heif_opener()
    HEIF_SUPPORT = True
except ImportError:
    HEIF_SUPPORT = False

# =========================
# CONFIGURATION
# =========================
INPUT_FOLDER = 'Testing_plat'
OUTPUT_FOLDER = 'Testing_plat_output'
PLATE_MODEL_PATH = os.getenv("PLATE_MODEL_PATH", "models/best_finetuned.pt")
YOLO_CONF_THRESHOLD = 0.15
MIN_PLATE_TEXT_LEN = 4

# =========================
# INITIALIZATION
# =========================
print(f"[INFO] Loading YOLO model from: {PLATE_MODEL_PATH}")
plate_model = YOLO(PLATE_MODEL_PATH)

print("[INFO] Initializing PaddleOCR...")
# PaddleOCR v2.7.3 API
reader = PaddleOCR(use_angle_cls=False, lang='en', show_log=False)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# =========================
# UTILITIES
# =========================
def clean_plate(text):
    """Remove non-alphanumeric characters and uppercase."""
    text = text.upper()
    text = re.sub(r'[^A-Z0-9]', '', text)
    return text

def extract_indonesian_plate(plate):
    """Mengekstrak HANYA bagian yang cocok dengan pola plat nomor Indonesia, mengabaikan teks sampah/tanggal pajak."""
    # Pola: 1-2 Huruf, diikuti 1-4 Angka, diikuti 0-3 Huruf
    match = re.search(r'([A-Z]{1,2}\d{1,4}[A-Z]{0,3})', plate)
    if match:
        return match.group(1)
    return None

def normalize_plate(plate):
    """Normalize plate text."""
    return plate.replace(" ", "").upper() if plate else ""

def load_image(filepath):
    """Load image, with HEIC support if available."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.heic' and HEIF_SUPPORT:
        img = Image.open(filepath)
        img = img.convert('RGB')
        open_cv_image = np.array(img)
        # Convert RGB to BGR
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        return open_cv_image
    else:
        return cv2.imread(filepath)

def run_paddle_ocr(img):
    """Run PaddleOCR on an image and return extracted text.
    PaddleOCR v2.7.3 returns: [[[box], (text, confidence)], ...]
    Input can be BGR (3-channel) or grayscale (1-channel)."""
    try:
        result = reader.ocr(img, cls=False)
        if not result or not result[0]:
            return ""
        texts = []
        for line in result[0]:
            # Each line: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], (text, confidence)
            if line and len(line) >= 2:
                text_info = line[1]
                if isinstance(text_info, tuple) and len(text_info) >= 1:
                    texts.append(text_info[0])
        return "".join(texts)
    except Exception as e:
        print(f"  [OCR ERROR] {e}")
        return ""

# =========================
# INFERENCE LOGIC
# =========================
def process_images():
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.heic')

    if not os.path.exists(INPUT_FOLDER):
        print(f"[ERROR] Folder {INPUT_FOLDER} tidak ditemukan.")
        return

    files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(valid_extensions)])
    print(f"[INFO] Ditemukan {len(files)} gambar di folder '{INPUT_FOLDER}'.")

    for idx, filename in enumerate(files, 1):
        filepath = os.path.join(INPUT_FOLDER, filename)
        print(f"\n[{idx}/{len(files)}] Memproses: {filename}")

        image = load_image(filepath)
        if image is None:
            print(f"  [WARNING] Gagal membaca {filename}. Melewati...")
            continue

        rendered_image = image.copy()

        # Inferensi YOLO
        results = plate_model(image)

        best_text = None
        best_conf = 0

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                if conf > YOLO_CONF_THRESHOLD:
                    plate_crop = image[y1:y2, x1:x2]
                    if plate_crop.size == 0:
                        continue

                    # === Pre-processing ===
                    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    
                    # 1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    gray_clahe = clahe.apply(gray)
                    clahe_bgr = cv2.cvtColor(gray_clahe, cv2.COLOR_GRAY2BGR)
                    
                    blur = cv2.GaussianBlur(gray_clahe, (5, 5), 0)

                    # 2. Otsu's Thresholding
                    _, otsu_thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    otsu_bgr = cv2.cvtColor(otsu_thresh, cv2.COLOR_GRAY2BGR)

                    # 3. Adaptive Thresholding
                    adapt_thresh = cv2.adaptiveThreshold(
                        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                    )
                    adapt_bgr = cv2.cvtColor(adapt_thresh, cv2.COLOR_GRAY2BGR)

                    # 4. Asli Resize
                    resized_crop = cv2.resize(plate_crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

                    # === OCR dengan fallback bertingkat ===
                    text_extracted = ""
                    candidates = []

                    # List urutan prioritas gambar pre-processing
                    prep_images = [otsu_bgr, adapt_bgr, clahe_bgr, resized_crop]

                    for img_prep in prep_images:
                        temp_text = run_paddle_ocr(img_prep)
                        temp_clean = clean_plate(temp_text)
                        if temp_clean:
                            candidates.append(temp_clean)
                            valid_part = extract_indonesian_plate(temp_clean)
                            if valid_part:
                                text_extracted = valid_part
                                break  # Langsung berhenti jika ketemu pola plat yang valid

                    # Jika semuanya tidak memiliki pola plat yang valid, fallback ke pembacaan terpanjang yang didapat
                    if not text_extracted and candidates:
                        text_extracted = max(candidates, key=len)

                    if len(text_extracted) >= MIN_PLATE_TEXT_LEN and conf > best_conf:
                        best_text = text_extracted
                        best_conf = conf

                    # === Render bounding box pada gambar ===
                    cv2.rectangle(rendered_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    display_text = text_extracted if text_extracted else "No Text"
                    label = f"{display_text} ({conf:.2f})"

                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(rendered_image, (x1, y1 - 25), (x1 + w + 4, y1), (0, 255, 0), -1)
                    cv2.putText(rendered_image, label, (x1 + 2, y1 - 7),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Normalize final plate text
        final_plate = normalize_plate(best_text)
        if final_plate:
            print(f"  [RESULT] Plat terdeteksi: {final_plate}")
        else:
            print("  [RESULT] Tidak ada plat yang terdeteksi dengan baik.")

        # Simpan gambar hasil render
        base_name = os.path.splitext(filename)[0]
        output_img_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_rendered.jpg")
        cv2.imwrite(output_img_path, rendered_image)

        # Simpan hasil teks ke .txt
        output_txt_path = os.path.join(OUTPUT_FOLDER, f"{base_name}.txt")
        with open(output_txt_path, "w") as f:
            f.write(final_plate if final_plate else "NOT_DETECTED")

    print(f"\n[INFO] Selesai memproses semua gambar. Hasil tersimpan di folder '{OUTPUT_FOLDER}'.")

if __name__ == '__main__':
    process_images()
