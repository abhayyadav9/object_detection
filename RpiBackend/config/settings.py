import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DETECTION_MODEL_PATH = os.getenv(
	"DETECTION_MODEL_PATH",
	os.path.join(os.path.dirname(BASE_DIR), "yolov10s.pt"),
)
POSE_MODEL_PATH = os.getenv(
	"POSE_MODEL_PATH",
	os.path.join(os.path.dirname(BASE_DIR), "yolov8s-pose.pt"),
)

CONF_THRESHOLD = float(os.getenv("DETECTION_CONF_THRESHOLD", "0.25"))
DETECTION_IOU_THRESHOLD = 0.45
DETECTION_IMAGE_SIZE = int(os.getenv("DETECTION_IMAGE_SIZE", "640"))
DETECTION_MAX_DET = int(os.getenv("DETECTION_MAX_DET", "50"))
DETECTION_AGNOSTIC_NMS = True
DETECTION_FALLBACK_PASSES = 1
DETECTION_TARGET_CLASSES = [
	item.strip().lower()
	for item in os.getenv(
		"DETECTION_TARGET_CLASSES",
		"person,car,bottle,laptop,bench,cell phone",
	).split(",")
	if item.strip()
]

POSE_CONF_THRESHOLD = 0.30
ACTIVITY_HISTORY_SIZE = 3

OCR_OEM = 3
OCR_PSM_VALUES = (6, 7, 11)
OCR_MIN_TEXT_SCORE = 0.35
OCR_ENABLED = os.getenv("OCR_ENABLED", "0") == "1"
OCR_LIGHTWEIGHT_ONLY = os.getenv("OCR_LIGHTWEIGHT_ONLY", "1") == "1"
