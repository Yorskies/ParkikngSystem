import os
import cv2
import sqlite3
from dotenv import load_dotenv

# Load fungsi dari face_engine
from face_engine import get_embedding_and_bbox, verify_embeddings

# 1. Load Environment Variables
load_dotenv()
FACE_CAMERA_URL = os.getenv("FACE_CAMERA_URL", "0")
if FACE_CAMERA_URL.isdigit():
    FACE_CAMERA_URL = int(FACE_CAMERA_URL)

DB_PATH = "database.db"

# 2. Load semua data wajah dari SQLite ke memori
def load_registered_faces():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    try:
        # Ambil user dan embeddingnya
        c.execute("""
            SELECT u.name, e.embedding 
            FROM users u 
            JOIN embeddings e ON u.id = e.user_id
        """)
        rows = c.fetchall()
    except sqlite3.OperationalError:
        print("[ERROR] Tabel tidak ditemukan. Pastikan database sudah terisi.")
        rows = []
    finally:
        conn.close()
    
    users_db = {}
    for row in rows:
        name = row[0]
        emb = row[1]
        if name not in users_db:
            users_db[name] = []
        users_db[name].append(emb)
        
    print(f"[INFO] Berhasil memuat data dari {len(users_db)} orang terdaftar.")
    return users_db

def main():
    print("[INFO] Menginisialisasi Model Wajah...")
    # Model wajah (InsightFace) otomatis di-load saat file diimport
    
    users_db = load_registered_faces()
    
    print(f"[INFO] Menghubungkan ke kamera: {FACE_CAMERA_URL}")
    cap = cv2.VideoCapture(FACE_CAMERA_URL)
    
    # Set buffer kecil agar live feed tidak delay (cocok untuk RTSP/IP Camera)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    
    if not cap.isOpened():
        print("[ERROR] Tidak dapat membuka kamera.")
        return
        
    print("[INFO] Kamera siap! Tekan 'q' pada jendela video untuk keluar.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Frame tidak terbaca. Menunggu...")
            cv2.waitKey(100)
            continue
            
        # Lakukan inferensi wajah (deteksi dan ambil vektor fitur)
        input_emb, bbox = get_embedding_and_bbox(frame)
        
        if bbox is not None and input_emb is not None:
            # Wajah terdeteksi, mari kita cari kecocokan di database
            x1, y1, x2, y2 = bbox
            
            best_overall_score = -1
            best_name = "Unknown"
            
            # Cocokkan dengan setiap pengguna di database
            for name, stored_embs in users_db.items():
                match, score = verify_embeddings(input_emb, stored_embs, threshold=0.4)
                
                if score > best_overall_score:
                    best_overall_score = score
                    if match:
                        best_name = name
            
            # Tentukan warna kotak (Hijau jika cocok, Merah jika Unknown)
            color = (0, 255, 0) if best_name != "Unknown" else (0, 0, 255)
            
            # Gambar Bounding Box Wajah
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Tuliskan Nama dan Persentase Kecocokan (Score)
            text = f"{best_name} ({best_overall_score:.2f})"
            cv2.putText(frame, text, (x1, max(y1-10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
        else:
            # Tidak ada wajah di frame
            cv2.putText(frame, "Tidak ada wajah", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            
        # Tampilkan Jendela GUI OpenCV
        # Karena ukuran gambar kamera mungkin besar (HD), kita resize khusus jendela ini agar muat di monitor
        display_frame = cv2.resize(frame, (1024, 768))
        cv2.imshow("Tes Pengenalan Wajah InsightFace", display_frame)
        
        # Keluar jika menekan tombol 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Bersihkan memori dan tutup kamera
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
