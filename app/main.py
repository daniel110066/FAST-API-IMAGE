from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from .image_processor import process_image
import io

app = FastAPI()

@app.post("/process-image")
async def process(file: UploadFile = File(...)):
    contents = await file.read()
    data = process_image(contents)
    processed_image = data["processed_image"]
    return StreamingResponse(io.BytesIO(processed_image), media_type="image/png")
