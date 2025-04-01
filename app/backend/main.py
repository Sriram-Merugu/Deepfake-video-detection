import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from fastapi import FastAPI, File, UploadFile
import shutil
import uvicorn
import os
from fastapi.middleware.cors import CORSMiddleware
# Initialize FastAPI app
app = FastAPI()

# Load deepfake detection model
model = load_model("deepfake_timit_detector_final.h5")

# Directory for storing uploaded videos
UPLOAD_DIR = "uploaded_videos"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# Function to extract frames
def extract_frames(video_path, frame_interval=30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frame_resized = cv2.resize(frame, (224, 224))
            frames.append(frame_resized)
        count += 1
    cap.release()
    return np.array(frames) if frames else None


# Predict function
def predict_video(video_path, model, threshold=0.7):
    frames = extract_frames(video_path, frame_interval=30)
    if frames is None:
        return {"error": "No valid frames extracted"}

    frames_preprocessed = tf.keras.applications.xception.preprocess_input(frames.astype(np.float32))

    # Batch predict
    preds = model.predict(frames_preprocessed, batch_size=32)
    avg_pred = np.mean(preds)

    label = "Fake" if avg_pred > threshold else "Real"
    return {"label": label, "score": round(float(avg_pred), 3)}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173"],  # Allow frontend requests
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# FastAPI route for video upload and deepfake detection
@app.post("/predict/")
async def predict_deepfake(file: UploadFile = File(...)):
    # Save uploaded video
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run prediction
    result = predict_video(file_path, model)

    # Cleanup uploaded video
    os.remove(file_path)
    print(result)
    return result

# Run FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# http://127.0.0.1:8000/docs