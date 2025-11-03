import cv2
import numpy as np
import pytesseract
from fastapi import FastAPI, UploadFile, File
from mrz.checker.td3 import TD3CodeChecker
from mrz.base.codeline_tds import TD3CodeLine
from pydantic import BaseModel

app = FastAPI(title="MRZ OCR API")

class MRZResponse(BaseModel):
    success: bool
    message: str | None = None
    mrz_raw: list[str] | None = None
    parsed: dict | None = None


# -----------------------------
# MRZ PREPROCESSING
# -----------------------------
def preprocess_mrz(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # remove noise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # increase contrast
    gray = cv2.equalizeHist(gray)

    # strong threshold
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # upscale for better OCR
    th = cv2.resize(th, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    return th


# -----------------------------
# MRZ REGION DETECTION
# -----------------------------
def detect_mrz_region(image):
    h = image.shape[0]
    mrz_height = int(h * 0.28)  # MRZ is bottom ~25–30%

    mrz_region = image[h - mrz_height : h]
    return mrz_region


# -----------------------------
# OCR MRZ
# -----------------------------
def ocr_mrz(img):
    config = (
        "--oem 1 --psm 6 "
        "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
    )
    text = pytesseract.image_to_string(img, config=config)
    return text.strip().split("\n")


# -----------------------------
# PARSE MRZ USING python-mrz
# -----------------------------
def parse_mrz(lines: list[str]):
    if len(lines) < 2 or not lines[0] or not lines[1]:
        return None, "MRZ lines unreadable"

    try:
        line1 = TD3CodeLine(lines[0])
        line2 = TD3CodeLine(lines[1])
        checker = TD3CodeChecker(lines[0], lines[1])
    except Exception as e:
        return None, f"Parsing error: {str(e)}"

    if not checker.valid:
        return None, "Invalid MRZ checksum"

    parsed = {
        "document_type": line1.document_type,
        "country": line1.issuing_country,
        "surname": line1.surname,
        "given_names": line1.given_names,
        "document_number": line2.document_number,
        "nationality": line2.nationality,
        "date_of_birth": line2.birth_date,
        "sex": line2.sex,
        "date_of_expiry": line2.expiry_date,
        "personal_number": line2.optional_data
    }

    return parsed, None


# -----------------------------
# FASTAPI ENDPOINT
# -----------------------------
@app.post("/mrz", response_model=MRZResponse)
async def read_mrz(file: UploadFile = File(...)):
    contents = np.frombuffer(await file.read(), np.uint8)
    img = cv2.imdecode(contents, cv2.IMREAD_COLOR)
    if img is None:
        return MRZResponse(success=False, message="Invalid image format")

    mrz_region = detect_mrz_region(img)
    preprocessed = preprocess_mrz(mrz_region)
    lines = ocr_mrz(preprocessed)

    # clean empty lines
    lines = [l for l in lines if l.strip()]

    if len(lines) < 2:
        return MRZResponse(
            success=False,
            message="MRZ not detected. Ensure you upload the BACK side of the ID."
        )

    parsed, err = parse_mrz(lines)

    if err:
        return MRZResponse(success=False, message=err, mrz_raw=lines)

    return MRZResponse(
        success=True,
        message="MRZ extracted successfully",
        mrz_raw=lines,
        parsed=parsed
    )
