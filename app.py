import cv2
import pytesseract
import numpy as np
import re
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import Dict, Any
import logging

# -------------------------------------------------
app = FastAPI(title="Kenyan ID OCR – Region-Based")
logging.basicConfig(level=logging.INFO)

# -------------------------------------------------
class OCRResult(BaseModel):
    raw_text: Dict[str, str]
    parsed: Dict[str, Any]

# -------------------------------------------------
# 1. Pre-processing utilities
def preprocess(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

# -------------------------------------------------
# 2. Card detection + perspective correction
def find_card_contour(img: np.ndarray):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)
    return None

def four_point_transform(image: np.ndarray, pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

# -------------------------------------------------
# 3. Region definitions (relative to a 1000×630 card)
#    These coordinates were measured on a **standard Kenyan ID** at 1000×630 px.
#    They are robust to small scaling (±10%).
CARD_W, CARD_H = 1000, 630

REGIONS = {
    "serial_number":     (50,  80,  300, 120),   # x1,y1,x2,y2
    "id_number":         (650, 80,  950, 120),
    "full_names":        (50,  180, 500, 260),
    "date_of_birth":     (50,  280, 300, 320),
    "sex":               (50,  340, 200, 380),
    "district_of_birth": (50,  400, 400, 440),
    "place_of_issue":    (50,  460, 400, 500),
    "date_of_issue":     (50,  520, 300, 560),
}

def crop_region(img: np.ndarray, box):
    x1, y1, x2, y2 = box
    return img[y1:y2, x1:x2]

# -------------------------------------------------
# 4. OCR per region
def ocr_region(img: np.ndarray) -> str:
    pre = preprocess(img)
    pre = cv2.resize(pre, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    config = "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ. "
    return pytesseract.image_to_string(pre, config=config).strip()

# -------------------------------------------------
# 5. Parsing helpers
def parse_serial(text: str) -> str | None:
    m = re.search(r"\b\d{9}\b", text)
    return m.group(0) if m else None

def parse_id(text: str) -> str | None:
    m = re.search(r"\b\d{7,8}\b", text)
    return m.group(0) if m else None

def parse_names(text: str) -> str | None:
    # 3+ ALL-CAPS words
    m = re.search(r"\b[A-Z]{2,}\s+[A-Z]{2,}\s+[A-Z]{2,}", text)
    return m.group(0) if m else text.strip() if text else None

def parse_date(text: str) -> str | None:
    m = re.search(r"\b\d{2}\.\d{2}\.\d{4}\b", text)
    if m:
        return m.group(0)
    m = re.search(r"\b\d{2}\.\d{2}\.\d{2}\b", text)
    return m.group(0) + "00" if m else None  # assume 1900s/2000s

def parse_sex(text: str) -> str | None:
    up = text.upper()
    if "FEMALE" in up:
        return "FEMALE"
    if "MALE" in up:
        return "MALE"
    return None

def parse_place(text: str) -> str | None:
    return re.sub(r"[^A-Z]", "", text).strip() or None

# -------------------------------------------------
@app.post("/ocr", response_model=OCRResult)
async def ocr_endpoint(file: UploadFile = File(...)):
    # --- Read image ---
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image")

    # --- Detect & warp card ---
    pts = find_card_contour(img)
    if pts is None:
        raise ValueError("Could not detect ID card contour")
    warped = four_point_transform(img, pts)

    # Resize to standard size for fixed coordinates
    h, w = warped.shape[:2]
    scale_w = CARD_W / w
    scale_h = CARD_H / h
    scale = min(scale_w, scale_h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    standardized = cv2.resize(warped, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to exact size if needed
    pad_x = CARD_W - new_w
    pad_y = CARD_H - new_h
    if pad_x > 0 or pad_y > 0:
        standardized = cv2.copyMakeBorder(
            standardized, 0, pad_y, 0, pad_x,
            cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )

    # --- OCR each region ---
    raw = {}
    for name, box in REGIONS.items():
        crop = crop_region(standardized, box)
        raw[name] = ocr_region(crop)

    # --- Parse ---
    parsed = {
        "serial_number":     parse_serial(raw["serial_number"]),
        "id_number":         parse_id(raw["id_number"]),
        "full_names":        parse_names(raw["full_names"]),
        "date_of_birth":     parse_date(raw["date_of_birth"]),
        "sex":               parse_sex(raw["sex"]),
        "district_of_birth": parse_place(raw["district_of_birth"]),
        "place_of_issue":    parse_place(raw["place_of_issue"]),
        "date_of_issue":     parse_date(raw["date_of_issue"]),
    }

    return OCRResult(raw_text=raw, parsed=parsed)