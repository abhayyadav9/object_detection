from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger
import cv2
import asyncio
import json
import time
import numpy as np
import base64

from backend.app.services.detection import detector
from backend.app.services.tracker import tracker
from backend.app.services.utils import draw_boxes, get_timestamp, setup_logging
from backend.app.config import config

router = APIRouter()

setup_logging()

@router.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time video streaming with object detection and tracking.
    Clients connect to this endpoint to send video frames and receive annotated frames back.
    """
    await websocket.accept()
    logger.info("WebSocket connection established for video stream.")

    frame_count = 0
    # last_detection_time = time.time() # This variable is not currently used after initial assignment.

    try:
        while True:
            # Receive video frame data from the frontend.
            # Expecting base64 encoded JPEG image data as a string.
            data = await websocket.receive_text()

            # Decode the base64 encoded image data into an OpenCV frame.
            try:
                img_bytes = base64.b64decode(data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    logger.warning("Could not decode frame from received data. Skipping frame.")
                    continue
            except Exception as e:
                logger.error(f"Error decoding frame: {e}. Data type: {type(data)}. Skipping frame.")
                continue

            frame_count += 1
            detections = []
            tracks = []

            # Strategy for performance: Run heavy object detector every N frames.
            # On intermediate frames, rely on the tracker to maintain object presence.
            if frame_count % config.DETECTOR_FRAME_INTERVAL == 0:
                start_detection_time = time.time()
                detections = await detector.detect(frame) # Perform object detection
                logger.info(f"Detection took: {time.time() - start_detection_time:.4f} seconds for frame {frame_count}")
                tracks = tracker.update(detections) # Update tracker with new detections
            else:
                # For frames where detector is not run, retrieve active tracks from the tracker.
                # A more advanced tracker would predict object positions in these intermediate frames.
                tracks = tracker.get_active_tracks()

            # Draw bounding boxes and labels on the frame for both detections and tracks.
            annotated_frame = draw_boxes(frame.copy(), detections, tracks)

            # Encode the annotated frame back to JPEG format for efficient streaming.
            _, buffer = cv2.imencode(".jpg", annotated_frame)
            
            # Send the JPEG encoded frame (as bytes) back to the client.
            await websocket.send_bytes(buffer.tobytes())

            # Log detections asynchronously to avoid blocking the video stream.
            # In a production setup, this could push to a message queue (e.g., Redis)
            # for background processing or use an asynchronous logging library.
            if detections or tracks: # Log if there are any detections or active tracks
                log_message = {
                    "timestamp": get_timestamp(),
                    "frame_count": frame_count,
                    "detections": detections,
                    "tracks": tracks
                }
                logger.info(f"Frame {frame_count} processed. Detections: {len(detections)}, Tracks: {len(tracks)}")
                # Potentially log full details to a separate system/queue for analysis if needed
                # logger.debug(f"Full log for frame {frame_count}: {log_message}")

    except WebSocketDisconnect:
        logger.info("WebSocket connection disconnected.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
