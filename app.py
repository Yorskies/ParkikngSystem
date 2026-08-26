import sqlite3
import numpy as np
import cv2
import re
import threading
import time
from flask import Flask, request, jsonify, render_template, Response
from ultralytics import YOLO
from paddleocr import PaddleOCR
import os
from dotenv import load_dotenv

load_dotenv()

# =========================
# CONFIG
# =========================
DB_PATH = "database.db"
UPLOAD_FOLDER = 'uploads/faces'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# URL Kamera IP DroidCam (Dari .env)
# Default menggunakan 0 jika tidak ada URL untuk jaga-jaga/testing lokal
PLATE_CAMERA_URL = os.getenv("PLATE_CAMERA_URL", "0")
FACE_CAMERA_URL = os.getenv("FACE_CAMERA_URL", "1")

# =========================
# IMPORT FACE ENGINE
# =========================
from face_engine import get_embedding, verify_embeddings, get_embedding_and_bbox

# =========================
# INIT APP
# =========================
app = Flask(__name__)

# =========================
# LOAD MODELS
# =========================
reader = PaddleOCR(use_angle_cls=False, lang='en', show_log=False)

import sys
import ultralytics
sys.modules['ultralytics.yolo'] = ultralytics

PLATE_MODEL_PATH = os.getenv("PLATE_MODEL_PATH", "models/best_finetuned.pt")
print(f"[INFO] Memuat model deteksi plat dari: {PLATE_MODEL_PATH}")
plate_model = YOLO(PLATE_MODEL_PATH)

# =========================
# THREAD LOCKS (Mencegah AI Crash karena diakses bersamaan)
# =========================
plate_lock = threading.Lock()
ocr_lock = threading.Lock()

# =========================
# DATABASE INIT
# =========================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            plate TEXT UNIQUE
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            embedding BLOB,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    conn.commit()
    conn.close()

# =========================
# GLOBAL VERIFICATION STATE
# =========================
VERIFICATION_STATE = {
    "step": "plate",
    "locked_plate": None,
    "locked_owner_id": None,
    "locked_owner_name": None,
    "face_attempts": 0,
    "last_active": time.time()
}

# =========================
# GLOBAL STATE
# =========================
DETECTION_ACTIVE = False

# =========================
# CAMERA STREAM CLASS (BACKGROUND THREAD)
# =========================
class CameraStream:
    def __init__(self, src=0, is_plate_cam=False, is_face_cam=False):
        if isinstance(src, str) and src.isdigit():
            src = int(src)
            
        self.src = src
        self.stream = None
        self.frame = None
        self.grabbed = False
        
        self.stopped = False
        self.last_access = time.time()
        
        self.is_plate_cam = is_plate_cam
        self.is_face_cam = is_face_cam
        self.latest_bboxes = []
        
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        
        if self.is_plate_cam or self.is_face_cam:
            self.detect_thread = threading.Thread(target=self.detect_loop, args=())
            self.detect_thread.daemon = True
            self.detect_thread.start()

    def detect_loop(self):
        global DETECTION_ACTIVE
        while True:
            if self.stopped:
                return
            if not DETECTION_ACTIVE or time.time() - self.last_access > 10:
                time.sleep(1)
                continue
                
            frame = self.frame
            if frame is not None:
                if self.is_plate_cam:
                    # Bypass/Pause YOLO jika sedang dalam tahap verifikasi wajah
                    if VERIFICATION_STATE["step"] != "plate":
                        self.latest_bboxes = []
                    else:
                        with plate_lock:
                            results = plate_model(frame, verbose=False)
                        bboxes = []
                        for result in results:
                            for box in result.boxes:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                conf = float(box.conf[0])
                                if conf > 0.15:
                                    bboxes.append((x1, y1, x2, y2, conf))
                        self.latest_bboxes = bboxes
                elif self.is_face_cam:
                    _, bbox = get_embedding_and_bbox(frame)
                    if bbox is not None:
                        # Format bbox: [x1, y1, x2, y2]
                        self.latest_bboxes = [(bbox[0], bbox[1], bbox[2], bbox[3], 1.0)]
                    else:
                        self.latest_bboxes = []
            time.sleep(0.05)  # Mencegah CPU usage 100%

    def update(self):
        global DETECTION_ACTIVE
        while True:
            if self.stopped:
                return
            
            # Jika mode deteksi mati atau tidak ada yang meminta frame, matikan kamera (Pause)
            if not DETECTION_ACTIVE or time.time() - self.last_access > 10:
                if self.stream is not None and self.stream.isOpened():
                    print(f"[CAM] Standby mode aktif. Menonaktifkan stream: {self.src}")
                    self.stream.release()
                    self.stream = None
                time.sleep(1) # Sleep lebih lama agar tidak boros CPU
                continue

            if self.stream is None or not self.stream.isOpened():
                print(f"[CAM] Mengaktifkan kembali stream: {self.src}...")
                self.stream = cv2.VideoCapture(self.src)
                self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                if not self.stream.isOpened():
                    time.sleep(2)
                continue

            if self.stream is not None and self.stream.isOpened():
                (grabbed, frame) = self.stream.read()
                if grabbed:
                    self.frame = frame  # Simpan frame dalam resolusi asli HD untuk OCR
                else:
                    self.stream.release()
                    self.stream = None
                
            time.sleep(0.01)

    def read(self):
        self.last_access = time.time()
        return self.frame

    def stop(self):
        self.stopped = True
        self.thread.join()
        self.stream.release()

# Inisialisasi Stream (Akan mulai otomatis berjalan di background)
print(f"[INFO] Menghubungkan ke Kamera Plat: {PLATE_CAMERA_URL}")
plate_cam = CameraStream(PLATE_CAMERA_URL, is_plate_cam=True)

print(f"[INFO] Menghubungkan ke Kamera Wajah: {FACE_CAMERA_URL}")
face_cam = CameraStream(FACE_CAMERA_URL, is_face_cam=True)

# =========================
# UTIL
# =========================
def normalize_plate(plate):
    return plate.replace(" ", "").upper() if plate else ""

def clean_plate(text):
    text = text.upper()
    text = re.sub(r'[^A-Z0-9]', '', text)
    return text

def extract_indonesian_plate(plate):
    """Mengekstrak HANYA bagian yang cocok dengan pola plat nomor Indonesia, mengabaikan teks sampah/tanggal pajak."""
    match = re.search(r'([A-Z]{1,2}\d{1,4}[A-Z]{0,3})', plate)
    if match:
        return match.group(1)
    return None

def run_paddle_ocr(img):
    try:
        with ocr_lock:
            result = reader.ocr(img, cls=False)
        if not result or not result[0]:
            return ""
        texts = []
        for line in result[0]:
            if line and len(line) >= 2:
                text_info = line[1]
                if isinstance(text_info, tuple) and len(text_info) >= 1:
                    texts.append(text_info[0])
        return "".join(texts)
    except Exception as e:
        print(f"[OCR ERROR] {e}")
        return ""

# =========================
# PLATE DETECTION + OCR
# =========================
def get_plate_text_auto(image):
    try:
        with plate_lock:
            results = plate_model(image)
        
        best_text = None
        best_conf = 0

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                if conf > 0.15:
                    plate_crop = image[y1:y2, x1:x2]
                    if plate_crop.size == 0:
                        continue

                    # Pre-processing
                    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    
                    # 1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    gray_clahe = clahe.apply(gray)
                    clahe_bgr = cv2.cvtColor(gray_clahe, cv2.COLOR_GRAY2BGR)
                    
                    blur = cv2.GaussianBlur(gray_clahe, (5, 5), 0)

                    # 2. Otsu
                    _, otsu_thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    otsu_bgr = cv2.cvtColor(otsu_thresh, cv2.COLOR_GRAY2BGR)

                    # 3. Adaptive
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

                    if len(text_extracted) >= 4 and conf > best_conf:
                        best_text = text_extracted
                        best_conf = conf

        if best_text:
            return normalize_plate(best_text)
        return None

    except Exception as e:
        print("[ERROR OCR]", e)
        return None

# =========================
# ROUTES UI & DATA
# =========================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/markets')
def markets():
    invert_camera = os.getenv("INVERT_CAMERA", "true").lower() == "true"
    return render_template('markets.html', invert_camera=invert_camera)

@app.route('/api/students', methods=['GET'])
def get_students():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT u.id, u.name, u.plate,
        (SELECT COUNT(id) FROM embeddings WHERE user_id = u.id)
        FROM users u
    """)
    rows = c.fetchall()
    conn.close()

    data = []
    for row in rows:
        data.append({
            "id": row[0],
            "name": row[1],
            "plate": row[2],
            "face_count": row[3]
        })
    return jsonify(data)

# =========================
# TOGGLE DETECTION API
# =========================
@app.route('/api/toggle_detection', methods=['POST'])
def toggle_detection():
    global DETECTION_ACTIVE
    data = request.json
    active = data.get("active", False)
    DETECTION_ACTIVE = active
    return jsonify({"status": "success", "active": DETECTION_ACTIVE})

# =========================
# LIVE VIDEO STREAMING (MJPEG)
# =========================
def generate_frames(camera):
    while True:
        frame = camera.read()
        if frame is None:
            time.sleep(0.1)
            continue
            
        render_frame = frame.copy()
        
        # Gambar overlay secara real-time berdasarkan memori terakhir dari thread
        if getattr(camera, 'is_plate_cam', False):
            for (x1, y1, x2, y2, conf) in camera.latest_bboxes:
                cv2.rectangle(render_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(render_frame, f"Plat: {conf:.2f}", (x1, max(y1-10, 0)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        elif getattr(camera, 'is_face_cam', False):
            for (x1, y1, x2, y2, conf) in camera.latest_bboxes:
                cv2.rectangle(render_frame, (x1, y1), (x2, y2), (255, 0, 0), 2) # Biru untuk wajah
                cv2.putText(render_frame, "Wajah", (x1, max(y1-10, 0)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                            
        # Perkecil resolusi khusus untuk video yang dikirim ke layar web agar tidak berat/lag
        render_frame_web = cv2.resize(render_frame, (640, 480))
            
        ret, buffer = cv2.imencode('.jpg', render_frame_web)
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed/plate')
def video_feed_plate():
    return Response(generate_frames(plate_cam), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/face')
def video_feed_face():
    return Response(generate_frames(face_cam), mimetype='multipart/x-mixed-replace; boundary=frame')

# =========================
# VERIFY LIVE (DIRECT FROM BACKEND CAMERAS)
# =========================
@app.route('/verify_live', methods=['GET'])
def verify_live():
    global VERIFICATION_STATE
    
    # Auto-Reset jika tidak ada aktivitas lebih dari 10 detik
    if time.time() - VERIFICATION_STATE["last_active"] > 10:
        VERIFICATION_STATE["step"] = "plate"
        VERIFICATION_STATE["locked_plate"] = None
        VERIFICATION_STATE["face_attempts"] = 0
        
    VERIFICATION_STATE["last_active"] = time.time()
    
    # Mengambil frame paling baru dari memory
    plate_frame = plate_cam.read()
    face_frame = face_cam.read()

    if plate_frame is None or face_frame is None:
        return jsonify({"status": "waiting", "msg": "Menunggu koneksi kamera..."})

    # =============================
    # STATE: MENCARI PLAT
    # =============================
    if VERIFICATION_STATE["step"] == "plate":
        plate_text = get_plate_text_auto(plate_frame)

        if not plate_text:
            return jsonify({
                "status": "scanning",
                "step": "plate",
                "name": "-",
                "plate": "-",
                "msg": "Plat tidak terdeteksi",
                "face_bbox": None
            })

        print(f"[LIVE] Plat terdeteksi: {plate_text}")

        # Cek Database
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT id, name FROM users WHERE plate=?", (plate_text,))
        owner_row = c.fetchone()
        conn.close()

        if not owner_row:
            print(f"[LIVE] Plat {plate_text} tidak terdaftar")
            return jsonify({
                "status": "fail",
                "step": "ownership",
                "name": "-",
                "plate": plate_text,
                "msg": f"Plat {plate_text} tidak terdaftar",
                "face_bbox": None
            })

        # KUNCI DATA DAN UBAH STATE KE PENCARIAN WAJAH
        VERIFICATION_STATE["step"] = "face"
        VERIFICATION_STATE["locked_plate"] = plate_text
        VERIFICATION_STATE["locked_owner_id"] = owner_row[0]
        VERIFICATION_STATE["locked_owner_name"] = owner_row[1]
        VERIFICATION_STATE["face_attempts"] = 0
        
        return jsonify({
            "status": "scanning",
            "step": "face",
            "name": owner_row[1],
            "plate": plate_text,
            "msg": f"Plat valid ({owner_row[1]}), posisikan wajah...",
            "face_bbox": None
        })

    # =============================
    # STATE: MENCARI WAJAH
    # =============================
    elif VERIFICATION_STATE["step"] == "face":
        owner_id = VERIFICATION_STATE["locked_owner_id"]
        owner_name = VERIFICATION_STATE["locked_owner_name"]
        plate_text = VERIFICATION_STATE["locked_plate"]
        
        # Fokus 100% pada wajah
        input_emb, face_bbox = get_embedding_and_bbox(face_frame)

        if input_emb is None:
            return jsonify({
                "status": "scanning",
                "step": "face",
                "name": owner_name,
                "plate": plate_text,
                "msg": f"Mencari wajah {owner_name}...",
                "face_bbox": None
            })

        # Verifikasi
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT embedding FROM embeddings WHERE user_id=?", (owner_id,))
        rows = c.fetchall()
        conn.close()

        stored_embeddings = [row[0] for row in rows]
        match, score = verify_embeddings(input_emb, stored_embeddings)

        print(f"[LIVE] Face match: {match}, score: {score:.4f}")

        if match:
            # Sukses! Reset state untuk mobil berikutnya
            VERIFICATION_STATE["step"] = "plate"
            VERIFICATION_STATE["locked_plate"] = None
            
            return jsonify({
                "status": "success",
                "step": "done",
                "name": owner_name,
                "plate": plate_text,
                "msg": "Akses diterima",
                "face_bbox": face_bbox,
                "score": round(float(score), 4)
            })
        else:
            VERIFICATION_STATE["face_attempts"] += 1
            if VERIFICATION_STATE["face_attempts"] >= 10:
                print(f"[LIVE] Wajah gagal diverifikasi 10x, reset ke pencarian plat.")
                VERIFICATION_STATE["step"] = "plate"
                VERIFICATION_STATE["locked_plate"] = None
                return jsonify({
                    "status": "fail",
                    "step": "face",
                    "name": owner_name,
                    "plate": plate_text,
                    "msg": "Batas waktu wajah habis. Scan plat ulang.",
                    "face_bbox": face_bbox,
                    "score": round(float(score), 4)
                })
                
            return jsonify({
                "status": "fail",
                "step": "face",
                "name": owner_name,
                "plate": plate_text,
                "msg": "Wajah tidak cocok!",
                "face_bbox": face_bbox,
                "score": round(float(score), 4)
            })

# =========================
# REGISTER
# =========================
@app.route('/register', methods=['POST'])
def register():
    name = request.form.get('name')
    plate = normalize_plate(request.form.get('plate'))
    files = request.files.getlist("images")

    if not name or not plate or len(files) == 0:
        return jsonify({"status": "fail", "msg": "Data tidak lengkap"})

    embeddings = []

    for idx, file in enumerate(files):
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            continue

        emb = get_embedding(image)
        if emb is not None:
            embeddings.append(emb)
            filename = f"{plate}_{idx}.jpg"
            cv2.imwrite(os.path.join(UPLOAD_FOLDER, filename), image)

    if len(embeddings) < 4:
        return jsonify({"status": "fail", "msg": f"Hanya {len(embeddings)} wajah terdeteksi (Min 4)."})

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("INSERT OR IGNORE INTO users (name, plate) VALUES (?, ?)", (name, plate))
        c.execute("SELECT id FROM users WHERE plate=?", (plate,))
        user_id = c.fetchone()[0]

        for emb in embeddings:
            c.execute(
                "INSERT INTO embeddings (user_id, embedding) VALUES (?, ?)",
                (user_id, emb.astype(np.float32).tobytes())
            )
        conn.commit()
        return jsonify({"status": "success", "msg": f"{len(embeddings)} wajah tersimpan"})
    except Exception as e:
        return jsonify({"status": "fail", "msg": str(e)})
    finally:
        conn.close()

# =========================
# DELETE
# =========================
@app.route('/delete_student/<int:user_id>', methods=['DELETE'])
def delete_student(user_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("DELETE FROM embeddings WHERE user_id=?", (user_id,))
        c.execute("DELETE FROM users WHERE id=?", (user_id,))
        conn.commit()
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "fail", "msg": str(e)})
    finally:
        conn.close()

# =========================
# MAIN
# =========================
if __name__ == '__main__':
    init_db()
    # PENTING: use_reloader=False karena kita pakai background threads untuk camera
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)