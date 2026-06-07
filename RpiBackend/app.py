import warnings
import json
warnings.filterwarnings('ignore')

from flask import Flask, request, jsonify
from flask_cors import CORS

from models.detection import load_detection_model
from models.pose import load_pose_model

from utils.preprocess import preprocess
from utils.pose_utils import analyze_pose
from utils.ocr_utils import clean_and_format_text
from utils.response_builder import build_response
from utils.helperfunctionOcr import extract_text_from_variants
from config.settings import OCR_ENABLED


from config.settings import CONF_THRESHOLD, DETECTION_IMAGE_SIZE, DETECTION_IOU_THRESHOLD
from config.settings import DETECTION_AGNOSTIC_NMS, DETECTION_MAX_DET, DETECTION_TARGET_CLASSES

app = Flask(__name__)
CORS(app)

# Load models
detection_model = load_detection_model()
pose_model      = load_pose_model()

models_loaded = detection_model is not None and pose_model is not None

print("\n🔥 YOLOv10 + OCR Server Ready on :5001")

@app.route("/detect", methods=["POST"])
def detect():

    if not models_loaded:
        return jsonify({"error": "Model load failed"}), 500

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img = preprocess(request.files["image"])
    if img is None:
        return jsonify({"error": "Unable to decode image"}), 400

    det_out = detection_model.predict(
        source=img,
        imgsz=DETECTION_IMAGE_SIZE,
        conf=CONF_THRESHOLD,
        iou=DETECTION_IOU_THRESHOLD,
        max_det=DETECTION_MAX_DET,
        agnostic_nms=DETECTION_AGNOSTIC_NMS,
        device="cpu",
        verbose=False,
    )[0]

    detections, persons = [], []

    for box in det_out.boxes:
        x1,y1,x2,y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = det_out.names[cls]
        label_key = str(label).strip().lower()

        if DETECTION_TARGET_CLASSES and label_key not in DETECTION_TARGET_CLASSES:
            continue

        detections.append({
            "label": label,
            "confidence": round(conf, 2),
            "bbox": [round(x1),round(y1),round(x2),round(y2)]
        })

        if label == "person" and conf >= CONF_THRESHOLD:
            persons.append([x1,y1,x2,y2])

    # Activity detection is expensive on CPU, so only run it when a person is found.
    activities = []
    if persons:
        for bbox in persons[:2]:
            x1,y1,x2,y2 = map(int, bbox)
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            pose_out = pose_model(crop, verbose=False)[0]
            if len(pose_out.keypoints):
                keypoints = pose_out.keypoints.data[0].cpu().numpy()
                activities.append({
                    "bbox": bbox,
                    "activity": analyze_pose(keypoints, bbox)
                })

    # OCR is optional because it is much slower than object detection on CPU.
    raw_text = ""
    meaning = None
    if OCR_ENABLED:
        texts = extract_text_from_variants(img)
        merged_text = " ".join(dict.fromkeys([t.strip() for t in texts if t.strip()]))
        raw_text = merged_text
        meaning = clean_and_format_text(raw_text)

    
    
    

    response = build_response(detections, persons, activities, raw_text, meaning)

    print(
        "[DETECT] objects={objects} persons={persons} text={text!r} meaning={meaning!r}".format(
            objects=len(detections),
            persons=len(persons),
            text=raw_text,
            meaning=meaning,
        )
    )
    print(json.dumps(response, ensure_ascii=False, default=str))

    return jsonify(response)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "running", "loaded": models_loaded})


if __name__ == "__main__":
    print("\n🔥 YOLOv10 Pose Server Running at http://0.0.0.0:5001")
    app.run(host="0.0.0.0", port=5001, debug=False)