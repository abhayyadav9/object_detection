import os
import uuid
import numpy as np
import cv2
import pytesseract

# Explicitly set Tesseract executable path for Windows environments.
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def preprocess(file):
    """
    Preserve existing API: decode an uploaded file-like object into a BGR image.
    """
    data = np.frombuffer(file.read(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def preprocess_image(img):
    """
    Generate multiple processed variants of the input image to improve OCR accuracy.

    Returns a list of processed images (numpy arrays):
    - Grayscale
    - Gaussian blur + OTSU threshold
    - Adaptive threshold (mean)
    - Morphological dilation on adaptive threshold
    - CLAHE + bilateral filter + adaptive Gaussian threshold (NEW)
    """
    if img is None:
        return []

    # Ensure image is in BGR format
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # --- Variant 1: Grayscale ---
    gray_img = gray

    # --- Variant 2: Gaussian blur + OTSU ---
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # --- Variant 3: Adaptive mean threshold ---
    adaptive_mean = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 10
    )

    # --- Variant 4: Dilated adaptive mean threshold ---
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilated = cv2.dilate(adaptive_mean, kernel, iterations=1)

    # --- Variant 5 (NEW): CLAHE + Bilateral Filter + Adaptive Gaussian Threshold ---
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)

    # Reduce noise but keep edges
    bilateral = cv2.bilateralFilter(clahe_img, d=9, sigmaColor=75, sigmaSpace=75)

    adaptive_gaussian = cv2.adaptiveThreshold(
        bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 8
    )

    return [gray_img, otsu, adaptive_mean, dilated, adaptive_gaussian]


def select_best_text(text_list):
    """
    Return the longest non-empty string from the list; otherwise an empty string.
    """
    if not text_list:
        return ""
    cleaned = [t.strip() for t in text_list if isinstance(t, str) and t.strip()]
    if not cleaned:
        return ""
    return max(cleaned, key=len)


def extract_text(img):
    """
    Run OCR on multiple preprocessed versions of the image and return the best result.
    Stores debug images under the `ocr_debug_images` directory.
    """
    if img is None:
        return ""

    processed_images = preprocess_image(img)
    if not processed_images:
        return ""

    # Prepare debug directory
    debug_dir = os.path.join(os.path.dirname(__file__), "..", "ocr_debug_images")
    debug_dir = os.path.abspath(debug_dir)
    os.makedirs(debug_dir, exist_ok=True)

    results = []
    for idx, proc in enumerate(processed_images):
        # Save debug variant
        debug_name = f"ocr_{uuid.uuid4().hex}_{idx}.png"
        debug_path = os.path.join(debug_dir, debug_name)
        cv2.imwrite(debug_path, proc)

        # Convert grayscale to RGB for pytesseract
        if len(proc.shape) == 2:
            rgb = cv2.cvtColor(proc, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)

        text = pytesseract.image_to_string(rgb)
        cleaned = text.strip()
        if cleaned:
            results.append(cleaned)

    return select_best_text(results)
