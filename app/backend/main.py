import os
import cv2
import torch
import numpy as np
from torchvision import transforms, models
from torch import nn
from fastapi import FastAPI, File, UploadFile
import shutil
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# -----------------------------
# Helper: YOLOv8 Face Extraction
# -----------------------------
from ultralytics import YOLO  # Ensure ultralytics is installed

# Load YOLOv8 model (nano version for speed)
yolo_model = YOLO('yolov8n.pt')

# Normalization parameters (must match training)
im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Define transformation for individual frames
frame_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])


def extract_face(frame_tensor):
    """
    Denormalize a frame tensor ([3,112,112]), convert it to a NumPy image,
    run YOLOv8 to detect a 'person' (COCO class 0), and return a face crop resized to 112x112.
    """
    # Denormalize and convert to uint8 image
    frame_np = frame_tensor.cpu().numpy().transpose(1, 2, 0)
    frame_np = (frame_np * np.array(std) + np.array(mean)) * 255.0
    frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
    frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

    results = yolo_model(frame_bgr, verbose=False)
    boxes = results[0].boxes
    H, W, _ = frame_bgr.shape
    best_box = None
    best_conf = 0
    for box in boxes:
        cls = int(box.cls.item())
        conf = box.conf.item()
        if cls == 0 and conf > best_conf:
            best_conf = conf
            best_box = box.xyxy[0].tolist()
    if best_box is not None:
        xmin, ymin, xmax, ymax = map(int, best_box)
        face_crop = frame_bgr[ymin:ymax, xmin:xmax]
        if face_crop.size == 0:
            face_crop = frame_bgr[H // 4:3 * H // 4, W // 4:3 * W // 4]
    else:
        face_crop = frame_bgr[H // 4:3 * H // 4, W // 4:3 * W // 4]
    face_crop = cv2.resize(face_crop, (im_size, im_size))
    face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    face_crop_tensor = transforms.ToTensor()(face_crop)
    return face_crop_tensor.to(frame_tensor.device)


# -----------------------------
# Model Components
# -----------------------------
class SimpleDiscriminator(nn.Module):
    def __init__(self):
        super(SimpleDiscriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(64, 128)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class HybridModel(nn.Module):
    def __init__(self, num_classes, bidirectional=False):
        super(HybridModel, self).__init__()
        # Branch 1: ResNeXt50 + LSTM branch
        weights = models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1
        backbone = models.resnext50_32x4d(weights=weights)
        self.cnn_backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.lstm = nn.LSTM(2048, 2048, 1, bidirectional=bidirectional, batch_first=True)
        self.fc_branch1 = nn.Linear(2048, num_classes)

        # Branch 2: Face-based branch
        self.face_cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.discriminator = SimpleDiscriminator()
        self.fc_branch2 = nn.Linear(192, num_classes)  # 64 + 128 = 192

        # Fusion layer: fuse outputs from both branches.
        self.fusion_fc = nn.Linear(num_classes * 2, num_classes)

    def forward(self, x, return_features=False):
        batch_size, seq_length, C, H, W = x.shape

        # Branch 1: Process full frame sequence.
        x1 = x.view(batch_size * seq_length, C, H, W)
        feat = self.cnn_backbone(x1)
        feat = self.avgpool(feat)
        feat = feat.view(batch_size, seq_length, 2048)
        lstm_out, _ = self.lstm(feat)
        branch1_out = self.fc_branch1(torch.mean(lstm_out, dim=1))

        # Branch 2: Process face-based features for each frame.
        branch2_out_list = []
        for i in range(batch_size):
            frame_feats = []
            for j in range(seq_length):
                frame = x[i, j]
                face = extract_face(frame)
                f_feat = self.face_cnn(face.unsqueeze(0))
                f_feat = f_feat.view(64)
                d_feat = self.discriminator(face.unsqueeze(0))
                d_feat = d_feat.view(128)
                combined_feat = torch.cat([f_feat, d_feat], dim=0)  # [192]
                frame_feats.append(combined_feat)
            aggregated_feat = torch.stack(frame_feats).mean(dim=0)
            branch2_out_i = self.fc_branch2(aggregated_feat)
            branch2_out_list.append(branch2_out_i)
        branch2_out = torch.stack(branch2_out_list)

        fusion_input = torch.cat([branch1_out, branch2_out], dim=1)
        out = self.fusion_fc(fusion_input)

        if return_features:
            return out, fusion_input
        else:
            return out


# -----------------------------
# Video Processing Functions
# -----------------------------
def extract_frames(video_path, num_frames=10):
    """
    Extract frames uniformly from the video. If the video has fewer than num_frames,
    duplicate the last frame until num_frames is reached.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Determine indices to sample uniformly
    if total_frames >= num_frames:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        indices = list(range(total_frames))

    current_frame = 0
    sampled_idx = 0
    target_idx = int(indices[sampled_idx]) if len(indices) > 0 else None

    while cap.isOpened() and target_idx is not None:
        ret, frame = cap.read()
        if not ret:
            break
        if current_frame == target_idx:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Apply transformation (resize, tensor, normalize)
            transformed = frame_transform(frame_rgb)
            frames.append(transformed)
            sampled_idx += 1
            if sampled_idx < len(indices):
                target_idx = int(indices[sampled_idx])
            else:
                target_idx = None
        current_frame += 1
    cap.release()

    # If no frames were extracted, raise an error.
    if len(frames) == 0:
        raise ValueError("No frames extracted from video")
    # If fewer frames were extracted, duplicate the last frame.
    while len(frames) < num_frames:
        frames.append(frames[-1])

    # Stack into a tensor of shape [num_frames, 3, 112, 112]
    return torch.stack(frames)


def predict_video(video_path, model, device):
    try:
        frames_tensor = extract_frames(video_path, num_frames=10)
    except Exception as e:
        return {"error": f"Frame extraction failed: {e}"}

    # Add batch dimension -> [1, seq_length, 3, 112, 112]
    input_tensor = frames_tensor.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        # Apply softmax to get probabilities
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        pred_class = int(np.argmax(probs))

    # Assuming label 1 means "Real" and 0 means "Fake"
    label = "Real" if pred_class == 1 else "Fake"
    return {"label": label, "probabilities": probs.tolist()}


# -----------------------------
# FastAPI App Setup
# -----------------------------
app = FastAPI()

# Allow CORS for local frontend (adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory to store uploaded videos temporarily
UPLOAD_DIR = "uploaded_videos"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize and load the hybrid model
num_classes = 2  # Real vs Fake
model = HybridModel(num_classes=num_classes).to(device)
# Load the saved state dict (update the path if needed)
model_checkpoint = "final_hybrid_model2.pt"
if os.path.exists(model_checkpoint):
    model.load_state_dict(torch.load(model_checkpoint, map_location=device))
    print("Hybrid model loaded successfully.")
else:
    print("Warning: Model checkpoint not found. Ensure the file exists.")


# -----------------------------
# API Endpoint for Prediction
# -----------------------------
@app.post("/predict/")
async def predict_deepfake(file: UploadFile = File(...)):
    # Save the uploaded video file temporarily
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run prediction on the video
    result = predict_video(file_path, model, device)

    # Clean up the uploaded video file
    os.remove(file_path)
    print("Prediction result:", result)
    return result


# -----------------------------
# Run the FastAPI app
# -----------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
