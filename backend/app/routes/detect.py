from fastapi import APIRouter, File, UploadFile, HTTPException
from loguru import logger
import cv2
import numpy as np
from typing import List, Dict, Any
import io

from backend.app.services.detection import detector
from backend.app.services.utils import setup_logging
from backend.app.config import config

router = APIRouter()

setup_logging() # Ensure logging is configured when this router is loaded

@router.post("/detect", tags=["Detection"], response_model=List[Dict[str, Any]], summary="Upload an image and get object detections")
async def detect_image(file: UploadFile = File(...)):
    """
    Uploads an image, performs object detection using the YOLOv8 model,
    and returns a list of detected objects with their bounding boxes, confidence scores, and class labels.

    Args:
        file (UploadFile): The image file to be uploaded for detection.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each dictionary represents a detected object.
                              Each object includes 'box' (xmin, ymin, xmax, ymax), 'confidence', and 'class'.

    Raises:
        HTTPException: If the uploaded file is not a valid image or detection fails.
    """
    logger.info(f"Received image for detection: {file.filename}")
    try:
        # Read the image file content
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Could not decode image.")

        # Perform object detection on the image frame
        detections = await detector.detect(frame)
        logger.info(f"Detected {len(detections)} objects in {file.filename}")

        return detections

    except Exception as e:
        logger.error(f"Error processing detection for {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process image for detection: {e}")
