from ultralytics import YOLO
from config.settings import POSE_MODEL_PATH

def load_pose_model():
    print("🔄 Loading YOLO Pose Model...")
    try:
        model = YOLO(POSE_MODEL_PATH)
        print("✅ Pose Model Loaded")
        return model
    except Exception as e:
        print("❌ Pose Model Load Error:", e)
        return None
