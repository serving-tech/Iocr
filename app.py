import cv2
import pytesseract
import numpy as np
from fastapi import FastAPI, UploadFile, File

app = FastAPI()

@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    text = pytesseract.image_to_string(img)

    return {"text": text}
