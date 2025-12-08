import pytesseract
import cv2
import platform

# Auto-detect OS → set tesseract path only on Windows
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)[1]

    text = pytesseract.image_to_string(gray)
    return text.strip()
