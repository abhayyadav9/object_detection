import cv2
import platform
import numpy as np
import pytesseract
from pytesseract import Output

from config.settings import OCR_MIN_TEXT_SCORE, OCR_PSM_VALUES

# Auto-detect OS → set tesseract path only on Windows
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text(img):
    if img is None:
        return ""

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    if min(gray.shape[:2]) < 600:
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )

    sharpen_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0],
    ])
    sharpened = cv2.filter2D(enhanced, -1, sharpen_kernel)

    variants = [enhanced, sharpened, otsu, adaptive]

    configs = [f"--oem 3 --psm {psm} -c preserve_interword_spaces=1" for psm in OCR_PSM_VALUES]

    scored_results = []

    for variant in variants:
        for config in configs:
            data = pytesseract.image_to_data(variant, output_type=Output.DICT, config=config)
            words = []
            confidences = []

            for word, confidence in zip(data.get("text", []), data.get("conf", [])):
                if not word or not word.strip():
                    continue
                words.append(word.strip())
                try:
                    conf_value = float(confidence)
                except (TypeError, ValueError):
                    continue
                if conf_value >= 0:
                    confidences.append(conf_value)

            text = " ".join(words).strip()
            if not text:
                continue

            confidence_score = sum(confidences) / len(confidences) if confidences else 0.0
            length_score = min(len(text) / 25.0, 2.0)
            scored_results.append((confidence_score + length_score, text))

    if scored_results:
        scored_results.sort(key=lambda item: item[0], reverse=True)
        return scored_results[0][1]

    fallback = pytesseract.image_to_string(enhanced, config="--oem 3 --psm 6")
    return fallback.strip()


def detect_text_regions(img):
    """
    Find likely text regions so OCR can run on smaller, higher-quality crops.
    Returns bounding boxes sorted top-to-bottom, left-to-right.
    """
    if img is None:
        return []

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    if min(gray.shape[:2]) < 600:
        gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    blackhat_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    blackhat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, blackhat_kernel)

    grad_x = cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=-1)
    grad_x = np.absolute(grad_x)
    grad_x = (255 * (grad_x / (grad_x.max() + 1e-6))).astype("uint8")

    grad_x = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3)))
    _, thresh = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)))
    thresh = cv2.erode(thresh, None, iterations=1)
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    h, w = enhanced.shape[:2]

    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        area = bw * bh
        if area < 500:
            continue
        if bw < 30 or bh < 12:
            continue
        aspect = bw / max(bh, 1)
        if aspect < 1.0 or aspect > 25.0:
            continue
        if bw > 0.95 * w or bh > 0.5 * h:
            continue
        boxes.append((x, y, x + bw, y + bh))

    boxes.sort(key=lambda box: (box[1], box[0]))
    return boxes


def ocr_text_score(text):
    if not text:
        return 0.0
    cleaned = text.strip()
    if not cleaned:
        return 0.0
    alpha_num = sum(ch.isalnum() for ch in cleaned)
    ratio = alpha_num / max(len(cleaned), 1)
    length_score = min(len(cleaned) / 50.0, 1.0)
    return (ratio * 0.6) + (length_score * 0.4)
