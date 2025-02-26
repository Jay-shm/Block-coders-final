import os
import cv2
import torch
import pickle
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from fastapi import WebSocket

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=False, device=device)
face_model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

model_path = "models/face_embeddings.pkl"
face_embeddings = {}

if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        face_embeddings = pickle.load(f)

async def recognize_faces(websocket: WebSocket):
    await websocket.accept()
    video = cv2.VideoCapture(0)

    try:
        while True:
            ret, frame = video.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, _ = mtcnn.detect(rgb_frame)

            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    face_img = rgb_frame[y1:y2, x1:x2]

                    if face_img.size == 0:
                        continue

                    face_img = cv2.resize(face_img, (160, 160))
                    face_img = torch.tensor(face_img).permute(2, 0, 1).float().div(255).sub(0.5).div(0.5).unsqueeze(0).to(device)

                    with torch.no_grad():
                        embedding = face_model(face_img).cpu().numpy().flatten()

                    best_match = "Unknown"
                    min_distance = float("inf")
                    threshold = 0.8

                    for stored_path, data in face_embeddings.items():
                        stored_name = data["name"]
                        stored_embedding = np.array(data["embedding"])
                        distance = np.linalg.norm(stored_embedding - embedding)

                        if distance < min_distance:
                            min_distance = distance
                            best_match = stored_name if distance < threshold else "Unknown"

                    await websocket.send_json({"name": best_match})

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as e:
        print(f"WebSocket Error: {e}")
    finally:
        video.release()
        cv2.destroyAllWindows()
        await websocket.close()
