import os
import cv2
import torch
import pickle
import numpy as np
import shutil
from fastapi import FastAPI, File, UploadFile, Form, WebSocket, WebSocketDisconnect
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
from PIL import Image
from typing import List

app = FastAPI()

dataset_path = 'datasets'
model_path = 'models/face_recognition.pkl'
os.makedirs(dataset_path, exist_ok=True)
os.makedirs('models', exist_ok=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=False, device=device)
face_recognition = InceptionResnetV1(pretrained='vggface2').eval().to(device)

transforms = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

face_embeddings = {}
if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        face_embeddings = pickle.load(f)

from fastapi import FastAPI, UploadFile, File, Form
import os
import shutil
import cv2
import torch
from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=False, device=device)

dataset_path = "datasets"

@app.post('/add_person')
async def add_person(name: str = Form(...), files: list[UploadFile] = File(...)):
    person_folder = os.path.join(dataset_path, name)
    os.makedirs(person_folder, exist_ok=True)
    
    image_count = 0
    for file in files:
        file_location = os.path.join(person_folder, file.filename)
        with open(file_location, 'wb') as f:
            shutil.copyfileobj(file.file, f)
        
        # Process image to extract face
        frame = cv2.imread(file_location)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb_frame)

        boxes, _ = mtcnn.detect(img_pil)
        
        if boxes is not None:
            for box in boxes:
                x, y, x2, y2 = map(int, box)
                face_img = frame[y:y2, x:x2]
                
                if face_img.size > 0:
                    face_filename = os.path.join(person_folder, f"{image_count}.jpg")
                    cv2.imwrite(face_filename, face_img)
                    image_count += 1
        
        os.remove(file_location)  # Remove original file after processing
    
    return {'message': f'{image_count} face images saved successfully'}

@app.post("/train")
def train_model():
    """
    Extract face embeddings from stored images and save them.
    """
    global face_embeddings
    face_embeddings = {}  # Reset existing embeddings

    print("Training model... Extracting face embeddings.")

    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_folder):
            continue

        print(f"Processing {person_name}...")

        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = transforms(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    embedding = face_recognition(img_tensor).cpu().numpy().flatten()

                face_embeddings[img_path] = {"name": person_name, "embedding": embedding}
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    with open(model_path, "wb") as f:
        pickle.dump(face_embeddings, f)

    return {"message": "Training complete. Embeddings saved."}

