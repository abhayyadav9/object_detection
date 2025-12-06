from ultralytics import YOLO
from config.settings import DETECTION_MODEL_PATH

def load_detection_model():
    print("🔄 Loading YOLOv10 Detection Model...")
    try:
        model = YOLO(DETECTION_MODEL_PATH)
        print("✅ Detection Model Loaded")
        return model
    except Exception as e:
        print("❌ Detection Model Load Error:", e)
        return None
