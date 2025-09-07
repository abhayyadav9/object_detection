from fastapi import APIRouter
from pydantic import BaseModel
from loguru import logger
import torch
import os

from backend.app.services.detection import detector
from backend.app.config import config
from backend.app.services.utils import setup_logging

router = APIRouter()

setup_logging() # Ensure logging is configured when this router is loaded

class StatusResponse(BaseModel):
    """
    Pydantic model for the response returned by the /status endpoint.
    Provides key information about the service's operational status and model configuration.
    """
    status: str
    message: str
    model_loaded: bool
    model_device: str
    cuda_available: bool
    yolo_model_path: str
    confidence_threshold: float

@router.get("/status", response_model=StatusResponse, tags=["Status"], summary="Get service and model status")
async def get_status():
    """
    HTTP GET endpoint to retrieve the current health and status of the object detection service.
    This includes information about the YOLO model's loading status, device in use,
    CUDA availability, model path, and detection confidence threshold.

    Returns:
        StatusResponse: A response model containing the operational status details.
    """
    logger.info("Status endpoint called.")
    
    # Check if the detector model is loaded and determine the device it's running on.
    model_loaded = detector.model is not None
    model_device = detector.device
    # Check for CUDA (GPU) availability on the system.
    cuda_available = torch.cuda.is_available()

    return StatusResponse(
        status="ok",
        message="Service is running and model status checked.",
        model_loaded=model_loaded,
        model_device=model_device,
        cuda_available=cuda_available,
        yolo_model_path=config.YOLO_MODEL_PATH,
        confidence_threshold=config.CONFIDENCE_THRESHOLD
    )
