import cv2
import pytesseract
import numpy as np
import re
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import Dict, Any

# -------------------------------------------------
app = FastAPI(title="Kenyan ID MRZ OCR – Back Side Only")

# -------------------------------------------------
class MRZResult(BaseModel):
    raw_mrz: str
    mrz_lines: Dict[str, str]
    parsed: Dict[str, Any]
    checks: Dict[str, bool]

# -------------------------------------------------
# 1. MRZ PREPROCESSING
def preprocess_mrz(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # mild denoise but preserve edges
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # blackhat to highlight dark characters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 3))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    enhanced = cv2.addWeighted(gray, 1.0, blackhat, 1.5, 0)

    # resize up for Tesseract
    enhanced = cv2.resize(enhanced, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    # adaptive threshold
    th = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 15
    )

    # slight closing to connect broken characters
    close = cv2.morphologyEx(th, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))

    return close

# -------------------------------------------------
# 2. Extract MRZ text
def extract_mrz(img: np.ndarray) -> str:
    # MRZ is always at bottom ~25%
    h, w = img.shape[:2]
    mrz_region = img[int(h * 0.70):h, 0:w]  # take approx bottom 30%

    pre = preprocess_mrz(mrz_region)

    config = "--oem 1 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
    raw = pytesseract.image_to_string(pre, config=config)
    raw = raw.replace(" ", "")
    return raw.strip()

# -------------------------------------------------
# 3. Clean and split MRZ
def clean_mrz(raw: str) -> str:
    raw = raw.upper()
    raw = re.sub(r"[^A-Z0-9<\n]", "", raw)
    return raw

def split_mrz(raw: str):
    raw = clean_mrz(raw)
    lines = [x for x in raw.splitlines() if x.strip()]

    if len(lines) >= 2 and len(lines[-1]) >= 40:
        return lines[-2], lines[-1]

    # fallback: concatenate everything, slice into 2×44
    merged = "".join(lines)
    merged = re.sub(r"[^A-Z0-9<]", "", merged)

    merged = merged.ljust(88, "<")[:88]

    return merged[:44], merged[44:88]

# -------------------------------------------------
# 4. MRZ CHECKSUM
def mrz_checksum(s: str) -> int:
    values = {c: i for i, c in enumerate("0123456789<")}
    for i, ch in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ", start=10):
        values[ch] = i
    weights = [7, 3, 1]
    total = 0
    for i, ch in enumerate(s):
        total += values.get(ch, 0) * weights[i % 3]
    return total % 10

# -------------------------------------------------
# 5. Parse TD3 MRZ (Kenyan ID back uses similar format)
def parse_td3(line1: str, line2: str):
    doc_type = line1[0:2]
    country = line1[2:5]

    names_field = line1[5:].rstrip("<")
    parts = names_field.split("<<")
    surname = parts[0].replace("<", " ").strip()
    given = parts[1].replace("<", " ").strip() if len(parts) > 1 else ""

    passport = line2[0:9]
    passport_cd = line2[9]
    nationality = line2[10:13]
    birth = line2[13:19]
    birth_cd = line2[19]
    sex = line2[20]
    expiry = line2[21:27]
    expiry_cd = line2[27]
    personal = line2[28:42]
    personal_cd = line2[42]
    final_cd = line2[43]

    checks = {
        "passport_ok": mrz_checksum(passport) == int(passport_cd),
        "birth_ok": mrz_checksum(birth) == int(birth_cd),
        "expiry_ok": mrz_checksum(expiry) == int(expiry_cd),
        "personal_ok": mrz_checksum(personal) == int(personal_cd),
        "final_ok": mrz_checksum(passport + passport_cd + birth + birth_cd + expiry + expiry_cd + personal + personal_cd) == int(final_cd),
    }

    return {
        "document_type": doc_type,
        "country": country,
        "surname": surname,
        "given_names": given,
        "document_number": passport,
        "nationality": nationality,
        "date_of_birth": birth,
        "sex": sex,
        "date_of_expiry": expiry,
        "personal_number": personal,
    }, checks

# -------------------------------------------------
@app.post("/mrz", response_model=MRZResult)
async def mrz_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image")

    raw = extract_mrz(img)
    l1, l2 = split_mrz(raw)

    parsed, checks = parse_td3(l1, l2)

    return MRZResult(
        raw_mrz=raw,
        mrz_lines={"line1": l1, "line2": l2},
        parsed=parsed,
        checks=checks
    )
