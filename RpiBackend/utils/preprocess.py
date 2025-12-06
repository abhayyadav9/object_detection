import numpy as np
import cv2

def preprocess(file):
    bytes = np.frombuffer(file.read(), np.uint8)
    return cv2.imdecode(bytes, cv2.IMREAD_COLOR)
