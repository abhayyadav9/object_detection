import time
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import torch
import os
from collections import deque
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# -----------------------------------------
# MODEL PATH CONFIG
# -----------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DETECTION_MODEL_PATH = os.path.join(BASE_DIR, "yolov10s.pt")
POSE_MODEL_PATH      = os.path.join(BASE_DIR, "yolov8s-pose.pt")

CONF_THRESHOLD = 0.45
POSE_CONF_THRESHOLD = 0.30
ACTIVITY_HISTORY_SIZE = 3

print("📁 Detection Model:", DETECTION_MODEL_PATH)
print("📁 Pose Model:", POSE_MODEL_PATH)

# -----------------------------------------
# LOAD MODELS
# -----------------------------------------

models_loaded = True

try:
    print("🔄 Loading YOLOv10 Detection Model...")
    detection_model = YOLO(DETECTION_MODEL_PATH)
    print("✅ Detection Model Loaded")
except Exception as e:
    print("❌ Detection Model Error:", e)
    detection_model = None
    models_loaded = False

try:
    print("🔄 Loading YOLOv10 Pose Model...")
    pose_model = YOLO(POSE_MODEL_PATH)
    print("✅ Pose Model Loaded")
except Exception as e:
    print("❌ Pose Model Error:", e)
    pose_model = None
    models_loaded = False

frame_history = deque(maxlen=ACTIVITY_HISTORY_SIZE)

# -----------------------------------------
# COCO KEYPOINTS
# -----------------------------------------

COCO_KEYPOINTS = [
    "nose","left_eye","right_eye","left_ear","right_ear",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle"
]

# -----------------------------------------
# UTILS
# -----------------------------------------

def preprocess(file):
    bytes = np.frombuffer(file.read(), np.uint8)
    return cv2.imdecode(bytes, cv2.IMREAD_COLOR)

def dist(a, b):
    if a is None or b is None: return 0
    return float(np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2))

def angle(a, b, c):
    if None in (a, b, c): return 0
    a, b, c = map(np.array, (a, b, c))
    ba = a - b
    bc = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))

def kp(keypoints, i):
    if keypoints is None: return None
    if i >= len(keypoints): return None
    x, y, conf = keypoints[i]
    return (float(x), float(y)) if conf > POSE_CONF_THRESHOLD else None

# -----------------------------------------
# POSE ACTIVITY RECOGNITION
# -----------------------------------------

def analyze_pose(keypoints, box):
    ls, rs = kp(keypoints, 5), kp(keypoints, 6)
    lh, rh = kp(keypoints, 11), kp(keypoints, 12)
    lk, rk = kp(keypoints, 13), kp(keypoints, 14)
    la, ra = kp(keypoints, 15), kp(keypoints, 16)

    activities = []

    # Standing (torso height check)
    if ls and rs and lh and rh:
        torso = abs(((ls[1]+rs[1])/2) - ((lh[1]+rh[1])/2))
        box_h = box[3] - box[1]
        if torso / box_h > 0.75:
            activities.append("standing")

    # Sitting
    if lk and rk and la and ra:
        a1 = angle(lh, lk, la)
        a2 = angle(rh, rk, ra)
        if 70 <= (a1+a2)/2 <= 120:
            activities.append("sitting")

    # Running / Walking (leg spread)
    if la and ra:
        spread = dist(la, ra) / max(1, (box[2]-box[0]))
        if spread > 0.5:
            activities.append("running")
        elif spread > 0.3:
            activities.append("walking")

    return activities[0] if activities else "standing"

# -----------------------------------------
# POSE WRAPPER
# -----------------------------------------

def detect_activities(img, bboxes):
    final = []
    for b in bboxes:
        x1,y1,x2,y2 = map(int, b)
        crop = img[y1:y2, x1:x2]
        
        out = pose_model(crop, verbose=False)[0]
        if len(out.keypoints) == 0: continue

        keypoints = out.keypoints.data[0].cpu().numpy()
        activity = analyze_pose(keypoints, b)

        details = []
        for i, name in enumerate(COCO_KEYPOINTS):
            if i < len(keypoints):
                x, y, conf = keypoints[i]
                if conf > POSE_CONF_THRESHOLD:
                    details.append({"joint": name, "x": float(x), "y": float(y), "confidence": float(conf)})

        final.append({
            "bbox": b,
            "activity": activity,
            "keypoints": details[:5],
            "keypoints_count": len(details)
        })
    return final

# -----------------------------------------
# MAIN DETECT ENDPOINT
# -----------------------------------------

@app.route("/detect", methods=["POST"])
def detect():
    if not models_loaded:
        return jsonify({"error": "Model load failed"}), 500

    if "image" not in request.files:
        return jsonify({"error": "Image missing"}), 400

    img = preprocess(request.files["image"])
    det_out = detection_model(img, verbose=False)[0]

    detections, persons = [], []

    for box in det_out.boxes:
        x1,y1,x2,y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = det_out.names[cls]

        detections.append({
            "label": label,
            "confidence": round(conf, 2),
            "bbox": [round(x1),round(y1),round(x2),round(y2)]
        })

        if label == "person" and conf > 0.6:
            persons.append([x1,y1,x2,y2])

    human_activity = detect_activities(img, persons)

    return jsonify({
        "success": True,
        "detections": detections,
        "human_activities": human_activity,
        "human_count": len(persons)
    })

# -----------------------------------------
# HEALTH
# -----------------------------------------

@app.route("/health")
def health():
    return jsonify({
        "status": "running",
        "detection_model": DETECTION_MODEL_PATH,
        "pose_model": POSE_MODEL_PATH,
        "loaded": models_loaded
    })

# -----------------------------------------
# RUN
# -----------------------------------------

if __name__ == "__main__":
    print("\n🔥 YOLOv10 Pose Server Running at http://0.0.0.0:5001")
    app.run(host="0.0.0.0", port=5001, debug=False)
