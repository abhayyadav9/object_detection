import numpy as np
from config.settings import POSE_CONF_THRESHOLD

COCO_KEYPOINTS = [
    "nose","left_eye","right_eye","left_ear","right_ear",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle"
]

def dist(a, b):
    if a is None or b is None: return 0
    return float(np.linalg.norm(np.array(a)-np.array(b)))

def angle(a, b, c):
    if None in (a, b, c): return 0
    a, b, c = map(np.array, (a, b, c))
    ba = a - b
    bc = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))

def kp(keypoints, i):
    if i >= len(keypoints): return None
    x, y, conf = keypoints[i]
    return (float(x), float(y)) if conf > POSE_CONF_THRESHOLD else None

def analyze_pose(keypoints, box):
    ls, rs = kp(keypoints, 5), kp(keypoints, 6)
    lh, rh = kp(keypoints, 11), kp(keypoints, 12)
    lk, rk = kp(keypoints, 13), kp(keypoints, 14)
    la, ra = kp(keypoints, 15), kp(keypoints, 16)

    # Standing
    if ls and rs and lh and rh:
        torso = abs(((ls[1]+rs[1])/2) - ((lh[1]+rh[1])/2))
        box_h = box[3] - box[1]
        if torso / box_h > 0.75:
            return "standing"

    # Sitting
    if lk and rk and la and ra:
        a1 = angle(lh, lk, la)
        a2 = angle(rh, rk, ra)
        if 70 <= (a1+a2)/2 <= 120:
            return "sitting"

    # Walking / Running
    if la and ra:
        spread = dist(la, ra) / max(1, (box[2]-box[0]))
        if spread > 0.5:
            return "running"
        elif spread > 0.3:
            return "walking"

    return "standing"
