import cv2
import os
import time
import sqlite3
import numpy as np
from dotenv import load_dotenv
from face_engine import app, get_embedding_and_bbox, cosine_similarity

# Load environment variables
load_dotenv()
FACE_CAMERA_URL = os.getenv("FACE_CAMERA_URL", "0")
DB_PATH = "database.db"

def identify_face(input_emb):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT u.name, e.embedding FROM users u JOIN embeddings e ON u.id = e.user_id")
        rows = c.fetchall()
        conn.close()
    except Exception as e:
        print(f"[ERROR] Database error: {e}")
        return "DB Error", 0.0

    if not rows:
        return "Database Kosong", 0.0

    best_score = -1
    best_name = "Tidak Dikenal"

    for name, emb_blob in rows:
        stored_emb = np.frombuffer(emb_blob, dtype=np.float32)
        score = cosine_similarity(input_emb, stored_emb)
        if score > best_score:
            best_score = score
            best_name = name

    if best_score > 0.5:
        return best_name, best_score
    return "Tidak Dikenal", best_score

print("[INFO] Memulai Face Detection Testing Script...")
print(f"[INFO] Mencoba terhubung ke kamera: {FACE_CAMERA_URL}")

# Buka stream kamera
cap = cv2.VideoCapture(FACE_CAMERA_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

if not cap.isOpened():
    print("[ERROR] Gagal membuka kamera wajah. Pastikan IP dan DroidCam aktif.")
    exit(1)

print("[INFO] Kamera terhubung! Tekan 'q' pada keyboard untuk keluar.")

fps_start_time = time.time()
fps_frame_count = 0
current_fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("[WARN] Gagal membaca frame dari kamera. Mencoba ulang...")
        time.sleep(1)
        cap = cv2.VideoCapture(FACE_CAMERA_URL)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        continue

    # Resize frame
    frame = cv2.resize(frame, (640, 480))
    
    # Hitung FPS
    fps_frame_count += 1
    if time.time() - fps_start_time > 1.0:
        current_fps = fps_frame_count
        fps_frame_count = 0
        fps_start_time = time.time()

    # Jalankan Deteksi Wajah menggunakan InsightFace
    embedding, bbox = get_embedding_and_bbox(frame)

    if bbox is not None and embedding is not None:
        x1, y1, x2, y2 = bbox
        
        # Identifikasi wajah dari database
        name, conf = identify_face(embedding)
        
        # Gambar kotak
        color = (0, 255, 0) if name != "Tidak Dikenal" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Label teks
        label = f"{name} ({conf:.2f})"
        cv2.rectangle(frame, (x1, y1 - 30), (x2, y1), color, cv2.FILLED)
        cv2.putText(frame, label, (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        cv2.putText(frame, "Tidak Ada Wajah", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Indikator FPS
    cv2.putText(frame, f"FPS: {current_fps}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Testing Deteksi Wajah (InsightFace)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Selesai.")
