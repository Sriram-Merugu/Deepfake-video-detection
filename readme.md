
---

# Deepfake Video Detection Project

This project implements a deepfake video detection system using a hybrid model. The model leverages both full-frame and face-based features to classify videos as "Real" or "Fake." The backend is built with FastAPI, YOLO and PyTorch, and the frontend is developed using React and Tailwind CSS.

## Table of Contents

- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Running the Backend](#running-the-backend)
- [Running the Frontend](#running-the-frontend)
- [Usage](#usage)
- [Notes](#notes)
- [License](#license)

## Overview

The project is designed to detect deepfake videos by extracting a fixed number of frames from an uploaded video and running them through a hybrid deep learning model. The model consists of two branches:

1. **Full-Frame Branch:** Uses a ResNeXt50 backbone followed by an LSTM layer to process spatial and temporal features.
2. **Face-Based Branch:** Uses a YOLOv8-based helper to extract face regions from each frame and processes them through a CNN and a simple discriminator network.

Both branches produce outputs that are fused together to generate the final prediction.

## Model Architecture

- **ResNeXt50 + LSTM Branch:** Extracts spatial features from entire frames and models temporal relationships using an LSTM.
- **Face-Based Branch:** Uses YOLOv8 to extract faces from video frames. The extracted face is processed by a CNN followed by a discriminator network, which captures local facial features.
- **Fusion Layer:** Combines the outputs from both branches to make a final classification.
- **Output:** The model outputs a label ("Real" or "Fake") along with the associated probabilities.

The model was trained using a dataset consisting of real and manipulated videos. The final model checkpoint is saved as `final_hybrid_model3.pt`.


## Installation

### Backend Dependencies

Make sure you have Python 3.8+ installed. Then, install the required Python packages.

```bash
# Navigate to the backend directory (if applicable)
cd backend

# Install the required Python packages
pip install -r requirements.txt
```



### Frontend Dependencies

Ensure you have Node.js (v14 or later) installed. Then, install the Node dependencies:

```bash
# Navigate to the frontend directory
cd frontend

# Install the required Node packages
npm install
```

## Running the Backend

The backend is built using FastAPI. To start the server, run:

```bash
# From the backend directory
uvicorn main_hybrid:app --reload
```

This will start the FastAPI server on `http://127.0.0.1:8000` with interactive API documentation available at `http://127.0.0.1:8000/docs`.

## Running the Frontend

The frontend is a React application styled with Tailwind CSS. To start the development server, run:

```bash
# From the frontend directory
npm run dev
```

This command will start the React development server. You can then navigate to the provided URL (typically `http://localhost:3000` or similar) to interact with the application.

## Usage

1. **Upload Video:** Use the frontend interface to upload a video file.
2. **Prediction:** The video is sent to the backend API endpoint (`/predict/`), which processes the video by:
   - Extracting 90 uniformly sampled frames.
   - Preprocessing frames (resizing, normalizing, etc.).
   - Running the hybrid deepfake detection model.
3. **Results:** The API returns a JSON response with the prediction label (Real/Fake) and the associated probability scores.

## Notes

- Ensure the YOLOv8 model weights (`yolov8n.pt`) and the hybrid model checkpoint (`final_hybrid_model3.pt`) are in the correct locations as specified in the code. You can generate .pt file by running the notebook in kaggle, and then copy paste the model in the backend directory. the yolov8n.pt file will be automatically generated when main.py is run.
- If you encounter any issues with frame extraction or prediction, verify that the video file is valid and the required libraries are properly installed.
- For further customization or debugging, refer to the logs printed by the backend.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


---

