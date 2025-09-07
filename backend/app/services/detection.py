import torch
from ultralytics import YOLO
from loguru import logger
from typing import List, Dict, Any
import os

from backend.app.config import config

class ObjectDetector:
    """
    Manages the YOLOv8 object detection model, including loading, device selection (GPU/CPU fallback),
    and performing inference on video frames.
    """
    def __init__(self):
        """Initializes the ObjectDetector by loading the model and setting up the device."""
        self.model = self._load_model()
        self.device = self._set_device()
        logger.info(f"ObjectDetector initialized on device: {self.device}")

    def _load_model(self) -> YOLO:
        """Loads the YOLOv8 model, prioritizing ONNX/TensorRT if available, otherwise PyTorch."""
        model_path = config.YOLO_MODEL_PATH
        logger.info(f"Attempting to load YOLO model from: {model_path}")

        try:
            model = YOLO(model_path)
            logger.success(f"Successfully loaded YOLO model from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load YOLO model from {model_path}: {e}")
            logger.info("Attempting to load default YOLOv8n PyTorch model.")
            try:
                model = YOLO("yolov8n.pt")  # Load default nano model
                logger.success("Successfully loaded default YOLOv8n PyTorch model.")
                return model
            except Exception as e_default:
                logger.critical(f"Failed to load default YOLOv8n model: {e_default}")
                raise RuntimeError("Could not load any YOLOv8 model. Exiting.")

    def _set_device(self) -> str:
        """Sets the device to GPU (CUDA) if available, otherwise falls back to CPU."""
        if torch.cuda.is_available():
            device = config.DEVICE
            logger.info(f"CUDA is available. Using device: {device}")
        else:
            device = "cpu"
            logger.warning("CUDA is not available. Falling back to CPU for inference.")
        return device

    async def detect(self, frame: Any) -> List[Dict[str, Any]]:
        """
        Performs object detection on a single frame using the loaded YOLOv8 model.

        Args:
            frame (Any): The input image frame (e.g., NumPy array or PIL Image).

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing a detected object
                                  with its bounding box, confidence, and class label.
        """
        if self.model is None:
            logger.error("Detection model not loaded.")
            return []

        # Perform inference using YOLOv8 model
        # Parameters include confidence threshold, IoU threshold for NMS, device, and half-precision.
        results = self.model(frame, 
                             conf=config.CONFIDENCE_THRESHOLD, 
                             iou=config.IOU_THRESHOLD, 
                             device=self.device,
                             half=config.HALF_PRECISION and self.device != "cpu", # Use half precision only if GPU and enabled
                             verbose=False) # Suppress verbose output from YOLO
        
        detections = []
        # Process detection results
        for r in results:
            # Iterate through bounding boxes, confidence scores, and class IDs
            for *xyxy, conf, cls in r.boxes.data.tolist():
                label = self.model.names[int(cls)] # Get class name from class ID
                detections.append({
                    "box": [int(x) for x in xyxy],  # Bounding box coordinates (xmin, ymin, xmax, ymax)
                    "confidence": round(conf, 2),   # Confidence score of the detection
                    "class": label                  # Predicted class label
                })
        return detections

detector = ObjectDetector() # Initialize the singleton detector instance
