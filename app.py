from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import easyocr
import numpy as np
from PIL import Image
import io

app = FastAPI()

reader = easyocr.Reader(['en'], gpu=False)

@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    if file.content_type.split('/')[0] != "image":
        raise HTTPException(400, "Only image uploads allowed")

    img_bytes = await file.read()
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    np_img = np.array(pil_img)

    results = reader.readtext(np_img)

    out = [{"text": t, "conf": float(c)} for (_, t, c) in results]
    return JSONResponse({"results": out})

@app.get("/")
def root():
    return {"status": "ok"}
