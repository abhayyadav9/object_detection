from fastapi import APIRouter, BackgroundTasks, HTTPException
from loguru import logger
from pydantic import BaseModel
from typing import Dict, Any

from backend.app.services.retrain import student_model_trainer
from backend.app.services.utils import setup_logging

router = APIRouter()

setup_logging() # Ensure logging is configured when this router is loaded

class RetrainResponse(BaseModel):
    """
    Pydantic model for the response returned by the /retrain endpoint.
    Provides status and a message about the retraining process.
    """
    status: str
    message: str
    model_path: str | None = None

async def _run_retraining_task():
    """
    Asynchronous background task to execute the student model retraining process.
    This function is called by FastAPI's BackgroundTasks to avoid blocking the API response.
    It logs the start and completion status of the retraining.
    """
    logger.info("Retraining task started in background.")
    try:
        result = await student_model_trainer.retrain_model()
        logger.info(f"Retraining task completed with status: {result['status']}")
    except Exception as e:
        logger.error(f"Error during background retraining task: {e}")

@router.post("/retrain", response_model=RetrainResponse, tags=["Retrain"], summary="Trigger student model retraining")
async def trigger_retrain(background_tasks: BackgroundTasks):
    """
    HTTP POST endpoint to trigger the self-training (pseudo-label generation and student model retraining) process.
    The actual retraining is offloaded to a background task to ensure the API remains responsive.
    Clients receive an immediate response indicating that retraining has been initiated.

    Args:
        background_tasks (BackgroundTasks): FastAPI dependency to run tasks in the background.

    Returns:
        RetrainResponse: A response model indicating that the retraining process has been started.
    """
    logger.info("Retrain endpoint called. Triggering background retraining task.")
    background_tasks.add_task(_run_retraining_task) # Add the retraining function to background tasks
    return RetrainResponse(
        status="processing",
        message="Retraining initiated in the background. Check logs for progress.",
        model_path=None
    )
