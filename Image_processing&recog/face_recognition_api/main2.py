import cv2
import torch
import pickle
import numpy as np
import requests
from facenet_pytorch import InceptionResnetV1, MTCNN

# Set device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load FaceNet model
model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
mtcnn = MTCNN(keep_all=True, device=device)

# Load stored face embeddings
model_path = "models/face_recognition.pkl"
with open(model_path, "rb") as f:
    face_embeddings = pickle.load(f)

# API URLs
base_url = "http://127.0.0.1:8000/"
mark_attendance_url = f"{base_url}mark_attendance/"

# Open webcam
video = cv2.VideoCapture(0)

print("Starting real-time face recognition... Press 'q' to exit.")

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

            # Preprocess face for model
            face_img = cv2.resize(face_img, (160, 160))
            face_img = torch.tensor(face_img).permute(2, 0, 1).float().div(255).sub(0.5).div(0.5).unsqueeze(0).to(device)

            with torch.no_grad():
                embedding = model(face_img).cpu().numpy().flatten()

            best_match = "Unknown"
            best_uid = None
            min_distance = float("inf")
            threshold = 0.8  

            # Compare with stored embeddings
            for stored_path, data in face_embeddings.items():
                stored_uid = data["name"]  
                stored_name = data["name"]
                stored_embedding = np.array(data["embedding"])

                distance = np.linalg.norm(stored_embedding - embedding)

                if distance < min_distance:
                    min_distance = distance
                    best_match = stored_name if distance < threshold else "Unknown"
                    best_uid = stored_uid if distance < threshold else None

            # Draw bounding box
            color = (0, 255, 0) if best_match != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, best_match, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # ✅ Only send attendance if a known person is detected
            if best_match != "Unknown":
                data = {
                    "UID": best_uid,  # roll_no of student
                    "ispresent": True,
                    "subject_code": "CS101"  # Change this dynamically based on subject
                }
                try:
                    response = requests.post(mark_attendance_url, json=data)
                    print(response.json())  # Debugging output
                except requests.exceptions.RequestException as e:
                    print(f"Error sending attendance: {e}")

    # ✅ Show frame **after processing all faces**
    cv2.imshow("Face Recognition", frame)

    # ✅ Exit loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
