import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DETECTION_MODEL_PATH = os.path.join(os.path.dirname(BASE_DIR), "yolov10s.pt")
POSE_MODEL_PATH      = os.path.join(os.path.dirname(BASE_DIR), "yolov8s-pose.pt")

CONF_THRESHOLD = 0.45
POSE_CONF_THRESHOLD = 0.30
ACTIVITY_HISTORY_SIZE = 3
