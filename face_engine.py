import numpy as np
from insightface.app import FaceAnalysis
from numpy.linalg import norm


app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=-1)  

def get_embedding(image):
    faces = app.get(image)

    if len(faces) == 0:
        return None

    largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return largest_face.embedding

def get_embedding_and_bbox(image):
    faces = app.get(image)

    if len(faces) == 0:
        return None, None

    largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    # Convert bbox to int list [x1, y1, x2, y2]
    bbox = [int(v) for v in largest_face.bbox]
    return largest_face.embedding, bbox

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

def verify_embeddings(input_emb, stored_embeddings, threshold=0.5):
    best_score = -1

    for emb in stored_embeddings:
        emb = np.frombuffer(emb, dtype=np.float32)
        score = cosine_similarity(input_emb, emb)

        if score > best_score:
            best_score = score

    return best_score > threshold, best_score