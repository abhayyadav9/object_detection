import cv2
import numpy as np
from models.ocr import extract_text


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

    variants = []

    # base
    variants.append(img)

    # sharpened
    sharp = sharpen_image(img)
    variants.append(sharp)

    # rotated originals
    variants.extend(auto_rotate_variants(img))

    # rotated sharpened
    variants.extend(auto_rotate_variants(sharp))

    results = []
    for variant in variants:
        text = extract_text(variant)
        if text and text.strip():
            results.append(text.strip())

    return results
