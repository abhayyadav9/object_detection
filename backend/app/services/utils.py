import os
import cv2
import numpy as np
from loguru import logger
from datetime import datetime
from typing import List, Dict, Any

from backend.app.config import config

def setup_logging():
    """
    Configures the Loguru logger to output messages to both the console (stderr)
    and a rotating file specified in the application's configuration.
    """
    os.makedirs(config.LOG_DIR, exist_ok=True)
    logger.remove() # Remove default handler
    logger.add(os.sys.stderr, level="INFO")  # Add console sink
    logger.add(config.LOG_FILE, rotation="10 MB", level="INFO") # Add file sink with rotation
    logger.info("Logging configured for console and file.")

def draw_boxes(frame: np.ndarray, detections: List[Dict[str, Any]], tracks: List[Dict[str, Any]] = None) -> np.ndarray:
    """
    Draws bounding boxes, labels, and confidence scores for detections,
    and unique IDs for tracks on a given video frame.

    Args:
        frame (np.ndarray): The input image frame (OpenCV format).
        detections (List[Dict[str, Any]]): A list of detected objects, each with
                                          "box", "confidence", and "class" keys.
        tracks (List[Dict[str, Any]], optional): A list of tracked objects, each with
                                                 "id", "box", and "class" keys. Defaults to None.

    Returns:
        np.ndarray: The frame with bounding boxes and labels drawn.
    """
    # Draw detections (green boxes)
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = det["class"]
        confidence = det["confidence"]
        color = (0, 255, 0)  # Green color for detection boxes
        text = f"{label}: {confidence:.2f}"
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2) # Draw rectangle
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2) # Draw label
    
    # Draw tracks (blue boxes with IDs)
    if tracks:
        for track in tracks:
            x1, y1, x2, y2 = map(int, track["box"])
            track_id = track["id"]
            label = track["class"]
            color = (255, 0, 0) # Blue color for track boxes
            text = f"ID: {track_id} {label}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
    return frame

def get_timestamp() -> str:
    """
    Generates and returns the current timestamp as a formatted string.

    Returns:
        str: The current timestamp in "YYYY-MM-DD HH:MM:SS" format.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
