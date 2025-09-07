import os

class Config:
    """
    Configuration class to manage various settings for the object detection backend.
    Settings are loaded from environment variables or default values.
    """
    # General Project Settings
    PROJECT_NAME: str = "Real-time Object Detection Backend"
    PROJECT_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"

    # YOLOv8 Model Configuration
    # Path to the YOLOv8 model weights (e.g., .pt, .onnx, or .engine)
    YOLO_MODEL_PATH: str = os.getenv("YOLO_MODEL_PATH", "backend/app/models/yolo/yolov8n.pt")
    # Confidence threshold for object detection. Detections below this are ignored.
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
    # IoU (Intersection over Union) threshold for Non-Maximum Suppression (NMS).
    IOU_THRESHOLD: float = float(os.getenv("IOU_THRESHOLD", "0.45"))
    # Device to run inference on (e.g., "cuda:0" for GPU, "cpu" for CPU).
    DEVICE: str = os.getenv("DEVICE", "cuda:0")
    # Enable half-precision (FP16) inference if GPU supports it.
    HALF_PRECISION: bool = os.getenv("HALF_PRECISION", "True").lower() == "true"
    # Interval (in frames) at which the heavy object detector runs.
    # On intermediate frames, only the tracker updates.
    DETECTOR_FRAME_INTERVAL: int = int(os.getenv("DETECTOR_FRAME_INTERVAL", "5"))

    # ByteTrack Tracker Configuration
    # Number of frames to keep a track alive after its last detection.
    TRACKER_BUFFER: int = int(os.getenv("TRACKER_BUFFER", "30"))
    # Assumed frames per second for the tracker.
    TRACKER_FPS: int = int(os.getenv("TRACKER_FPS", "30"))

    # Retraining Configuration (Pseudo-labeling and Student Model Training)
    # Minimum confidence for a detection to be considered a pseudo-label.
    PSEUDO_LABEL_CONFIDENCE: float = float(os.getenv("PSEUDO_LABEL_CONFIDENCE", "0.7"))
    # Directory for human-labeled ground truth data.
    LABELED_DATA_DIR: str = "data/labeled"
    # Directory for raw unlabeled frames collected from the stream.
    UNLABELED_DATA_DIR: str = "data/unlabeled"
    # Directory to store auto-generated pseudo-labels.
    PSEUDO_LABELS_DIR: str = "data/pseudo_labels"
    # Number of epochs for student model retraining.
    RETRAIN_EPOCHS: int = int(os.getenv("RETRAIN_EPOCHS", "50"))
    # Batch size for student model retraining.
    RETRAIN_BATCH_SIZE: int = int(os.getenv("RETRAIN_BATCH_SIZE", "16"))
    # Image size for student model retraining.
    RETRAIN_IMG_SIZE: int = int(os.getenv("RETRAIN_IMG_SIZE", "640"))

    # Logging Configuration
    # Directory where log files will be stored.
    LOG_DIR: str = "logs"
    # Full path to the main detection log file.
    LOG_FILE: str = os.path.join(LOG_DIR, "detections.log")

    # Redis Configuration (for asynchronous task queuing, if implemented)
    # Hostname or IP address of the Redis server.
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    # Port number for the Redis server.
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))

config = Config()
