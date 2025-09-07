import pytest
import numpy as np
from unittest.mock import AsyncMock, patch

# Assuming detector and other dependencies can be mocked or initialized for testing
from backend.app.services.detection import ObjectDetector

@pytest.fixture
def mock_detector():
    with patch('ultralytics.YOLO') as mock_yolo_class:
        # Mock the YOLO model's __init__ and its predict method
        mock_model_instance = AsyncMock()
        mock_model_instance.names = {0: "person", 1: "bicycle"}
        
        # Mock the results object with a .boxes.data.tolist()
        mock_results = AsyncMock()
        mock_results.boxes.data.tolist.return_value = [
            [10, 10, 50, 50, 0.9, 0],  # person
            [60, 60, 100, 100, 0.8, 1] # bicycle
        ]
        mock_model_instance.return_value = lambda *args, **kwargs: [mock_results] # Mock the call to the model

        mock_yolo_class.return_value = mock_model_instance
        
        detector = ObjectDetector()
        yield detector

@pytest.mark.asyncio
async def test_object_detection(mock_detector):
    # Create a dummy frame (e.g., a black image)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    detections = await mock_detector.detect(dummy_frame)
    
    assert len(detections) == 2
    assert detections[0]["class"] == "person"
    assert detections[0]["confidence"] == 0.9
    assert detections[1]["class"] == "bicycle"
    assert detections[1]["confidence"] == 0.8

# Placeholder for other detection-related tests
# def test_gpu_fallback():
#     pass
