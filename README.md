# Real-time Video Streaming and Object Detection Backend

This project implements a real-time video streaming and object detection backend service with self-training capabilities, built with FastAPI, WebSockets, OpenCV, and Ultralytics YOLOv8.

## Folder Structure

```
backend/
 ├── app/
 │   ├── __init__.py
 │   ├── main.py                # FastAPI entry point
 │   ├── routes/
 │   │   ├── __init__.py
 │   │   ├── stream.py          # WebSocket for video
 │   │   ├── retrain.py         # trigger retraining
 │   │   ├── logs.py            # live detection logs
 │   │   ├── status.py          # health checks
 │   ├── services/
 │   │   ├── detection.py       # YOLOv8 detection logic
 │   │   ├── tracker.py         # ByteTrack/DeepSORT tracking
 │   │   ├── retrain.py         # pseudo-label generator + student retrain
 │   │   ├── utils.py           # helpers
 │   ├── models/
 │   │   ├── yolo/              # YOLO weights, ONNX/TensorRT engine
 │   │   ├── classifier/        # age/gender/extra classifiers
 │   └── config.py              # config vars
 ├── data/
 │   ├── labeled/               # human labeled data
 │   ├── unlabeled/             # raw frames
 │   ├── pseudo_labels/         # auto-generated labels
 ├── scripts/
 │   ├── collect_frames.py      # frame collection from stream
 │   ├── export_trt.py          # export YOLO → ONNX → TensorRT
 │   ├── train_student.py       # retraining loop
 ├── tests/
 │   ├── test_api.py
 │   ├── test_detection.py
 ├── requirements.txt
 ├── README.md
 ├── Dockerfile
 └── docker-compose.yml
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd detection_Backend
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Running Locally

To run the application locally, you can use Uvicorn:

```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Running with Docker

1.  **Build the Docker image:**
    ```bash
    docker-compose build
    ```

2.  **Run the Docker containers:**
    ```bash
    docker-compose up
    ```
    For GPU support, ensure you have NVIDIA Container Toolkit installed and configured.

## Deployment to Cloud

(Instructions for cloud deployment will be added here.)

## API Endpoints

-   `/stream`: WebSocket endpoint for live video streaming with object detection overlay.
-   `/logs`: HTTP endpoint to retrieve live detection logs.
-   `/retrain`: HTTP endpoint to trigger pseudo-label generation and student model retraining.
-   `/status`: HTTP endpoint for model health and status checks.
