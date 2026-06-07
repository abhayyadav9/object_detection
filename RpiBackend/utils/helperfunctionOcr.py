import cv2
import numpy as np
from models.ocr import extract_text, detect_text_regions, ocr_text_score
from config.settings import OCR_ENABLED, OCR_LIGHTWEIGHT_ONLY


def auto_rotate_variants(img):
    """
    Generate rotated versions of the image to improve OCR accuracy.
    Helps with tilted text (common in CCTV / handheld images).
    """
    angles = [-15, 0, 15]
    rotated = []
    h, w = img.shape[:2]

    for angle in angles:
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        rotated_img = cv2.warpAffine(img, M, (w, h))
        rotated.append(rotated_img)

    return rotated


def upscale_image(img, scale=2.0):
    """
    Upscale small text regions to improve OCR readability.
    """
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)


def sharpen_image(img):
    """
    Improve clarity using a sharpening kernel.
    Helps OCR detect characters more clearly.
    """
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    return cv2.filter2D(img, -1, kernel)


def extract_text_from_variants(img):
    """
    Run OCR on multiple enhanced versions of the same image:
    - original
    - sharpened
    - rotated (-15, 0, +15)
    - sharpened rotated variants

    Returns a list of all extracted text chunks.
    """

    if img is None:
        return []

    if not OCR_ENABLED:
        return []

    variants = []

    # base
    variants.append(img)

    # upscaled base only when not in lightweight mode
    if not OCR_LIGHTWEIGHT_ONLY:
        variants.append(upscale_image(img))

    # sharpened
    sharp = sharpen_image(img)
    variants.append(sharp)

    # sharpened + upscaled
    if not OCR_LIGHTWEIGHT_ONLY:
        variants.append(upscale_image(sharp))

    # rotated originals
    variants.extend(auto_rotate_variants(img))

    # rotated sharpened
    variants.extend(auto_rotate_variants(sharp))

    # rotated upscaled
    if not OCR_LIGHTWEIGHT_ONLY:
        variants.extend(auto_rotate_variants(upscale_image(img)))

    results = []
    for variant in variants:
        text = extract_text(variant)
        if text and text.strip():
            results.append(text.strip())

    region_results = []
    for x1, y1, x2, y2 in detect_text_regions(img):
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        text = extract_text(crop)
        if text and text.strip():
            region_results.append(text.strip())

    combined = results + region_results
    filtered = []
    for text in combined:
        if ocr_text_score(text) >= 0.35:
            filtered.append(text)

    return filtered
