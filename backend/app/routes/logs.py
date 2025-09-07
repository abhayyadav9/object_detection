
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger
import os
import asyncio

from backend.app.config import config
from backend.app.services.utils import setup_logging

router = APIRouter()

setup_logging() # Ensure logging is configured when this router is loaded

async def log_generator():
    """
    Asynchronous generator that continuously reads and yields new lines
    from the detection log file. It first yields existing content and then
    waits for and yields new lines as they are appended.

    Raises:
        HTTPException: If the log file does not exist.

    Yields:
        str: A new line from the log file, stripped of leading/trailing whitespace,
             followed by a newline character.
    """
    if not os.path.exists(config.LOG_FILE):
        logger.error(f"Log file not found at {config.LOG_FILE}")
        raise HTTPException(status_code=404, detail="Log file not found.")

    logger.info(f"Starting log stream from {config.LOG_FILE}")
    with open(config.LOG_FILE, "r") as f:
        # Initially send all existing content of the log file
        for line in f:
            yield line.strip() + "\n"

        # Continuously monitor for and send new lines appended to the log file
        while True:
            line = f.readline()
            if line:
                yield line.strip() + "\n"
            else:
                # If no new line, wait for a short period before checking again
                await asyncio.sleep(1) # Adjust sleep time based on expected log frequency

@router.get("/logs", tags=["Logs"], summary="Stream real-time detection logs")
async def get_detection_logs():
    """
    HTTP GET endpoint that provides a continuous stream of real-time object detection logs.
    The logs are streamed as `text/event-stream`, suitable for client-side EventSource API.

    Returns:
        StreamingResponse: A response object that streams log entries.

    Raises:
        HTTPException: If there's an issue setting up or maintaining the log stream.
    """
    try:
        return StreamingResponse(log_generator(), media_type="text/event-stream")
    except HTTPException as e:
        raise e # Re-raise HTTPExceptions (e.g., 404 for file not found)
    except Exception as e:
        logger.error(f"Error setting up log stream: {e}")
        raise HTTPException(status_code=500, detail="Failed to start log stream due to internal server error.")
