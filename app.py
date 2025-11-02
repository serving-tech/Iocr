import cv2
import pytesseract
import numpy as np
import re
from fastapi import FastAPI, UploadFile, File

app = FastAPI()


# ---------------------------------------------------------
# KENYAN ID OCR PARSER
# ---------------------------------------------------------
def parse_kenyan_id(text: str):
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    joined = " ".join(lines)

    def extract(pattern):
        match = re.search(pattern, joined, re.IGNORECASE)
        return match.group(1).strip() if match else None

    parsed = {
        "serial_number": extract(r"SERIAL NUMBER[:\s]+([A-Za-z0-9]+)"),
        "id_number": extract(r"(?:NUMBER|NO)[:\s]+([0-9]{5,})"),
        "full_names": extract(r"FULL NAMES\s+([A-Z\s]+)"),
        "date_of_birth": extract(r"DATE OF BIRTH\s+([0-9./-]+)"),
        "sex": extract(r"SEX\s+([A-Z]+)"),
        "district_of_birth": extract(r"DISTRICT OF BIRTH\s+([A-Z]+)"),
        "place_of_issue": extract(r"PLACE OF ISSUE\s+([A-Z]+)"),
        "date_of_issue": extract(r"DATE OF ISSUE\s+([0-9./-]+)"),
        "holder_signature": extract(r"HOLDER'S SIGN\.?\s*([A-Za-z.]+)")
    }

    # remove empty or None fields
    parsed = {k: v for k, v in parsed.items() if v}

    return parsed


# ---------------------------------------------------------
# OCR ENDPOINT
# ---------------------------------------------------------
@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    img_bytes = await file.read()

    # Load image
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Invalid image format"}

    # OCR text
    text = pytesseract.image_to_string(img)

    # Parse ID fields
    parsed = parse_kenyan_id(text)

    return {
        "raw_text": text,
        "parsed": parsed
    }
