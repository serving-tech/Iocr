import cv2
import pytesseract
import numpy as np
import re
from fastapi import FastAPI, UploadFile, File

app = FastAPI()

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    gray = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)[1]
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    return gray

def extract_fields(text):
    out = {}

    # Serial number (mostly 9 digits)
    m = re.search(r"\b(\d{9})\b", text)
    if m:
        out["serial_number"] = m.group(1)

    # ID number (7–8 digits usually)
    m = re.search(r"\b(\d{7,8})\b", text)
    if m:
        out["id_number"] = m.group(1)

    # Full names (ALL CAPS words)
    m = re.search(r"\b([A-Z]{3,}\s+[A-Z]{3,}\s+[A-Z]{3,})\b", text)
    if m:
        out["full_names"] = m.group(1)

    # Date of birth
    m = re.search(r"\b(\d{2}\.\d{2}\.\d{2,4})\b", text)
    if m:
        out["date_of_birth"] = m.group(1)

    # Sex
    if "FEMALE" in text.upper():
        out["sex"] = "FEMALE"
    elif "MALE" in text.upper():
        out["sex"] = "MALE"

    # District of birth (line after “DISTRICT OF BIRTH”)
    m = re.search(r"DISTRICT OF BIRTH\s*\n*([A-Z]+)", text, re.IGNORECASE)
    if m:
        out["district_of_birth"] = m.group(1).upper()

    # Place of issue
    m = re.search(r"PLACE OF ISSUE\s*\n*([A-Z]+)", text, re.IGNORECASE)
    if m:
        out["place_of_issue"] = m.group(1).upper()

    # Date of issue
    m = re.search(r"DATE OF ISSUE\s*\n*(\d{2}\.\d{2}\.\d{2,4})", text, re.IGNORECASE)
    if m:
        out["date_of_issue"] = m.group(1)

    return out

@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    processed = preprocess(img)

    text = pytesseract.image_to_string(
        processed,
        config="--oem 3 --psm 6"
    )

    parsed = extract_fields(text.upper())

    return {
        "raw_text": text,
        "parsed": parsed
    }
