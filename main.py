from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import os
import asyncio
import tempfile
import easyocr
import cv2
import re
app = FastAPI()
model = None  # Will be loaded during startup

VEHICLE_CLASS_MAP = {
    "car": "mobil",
    "ambulance": "mobil",
    "van": "mobil",
    "pickup truck": "mobil",
    "taxi": "mobil",
    "jeep": "mobil",
    "motorcycle": "motor",
    "scooter": "motor",
    "bus": "bis",
    "truck": "truk",
    "fire engine": "truk",
}

@app.on_event("startup")
async def startup_event():
    # Load the model once at startup
    global model, model_plate, reader, CLASS_NAMES
    model = YOLO("model/yolov8n.onnx")
    model_plate = YOLO("model/plate_model.onnx")
    CLASS_NAMES = model.names
    reader = easyocr.Reader(['en'])
@app.on_event("shutdown")
async def shutdown_event():
    # Optionally, unload the model or perform other cleanup tasks
    global model, model_plate
    model = None
    model_plate = None
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Use the tempfile module to safely create a temporary file
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            temp_file_path = tmp.name
            # Read the file asynchronously and write it synchronously
            content = await file.read()
            tmp.write(content)
        # Execute model inference in a separate thread to avoid blocking the event loop
        vehicle_results, plate_results = await asyncio.gather(
            asyncio.to_thread(model, temp_file_path),
            asyncio.to_thread(model_plate, temp_file_path)
        )
        # Process results
        vehicle_predictions = []
        for result in vehicle_results:
            for box in result.boxes:
                label_idx = int(box.cls[0])
                label_en = CLASS_NAMES[label_idx]
                label_id = VEHICLE_CLASS_MAP.get(label_en)
                if label_id is None:
                    continue
                vehicle_predictions.append({
                    "class": label_id,
                    "confidence": float(box.conf[0]),
                    "coordinates": box.xyxy.tolist()
                })

        # Process plate detections
        plate_predictions = []
        for result in plate_results:
            for box in result.boxes:
                plate_predictions.append({
                    "confidence": float(box.conf[0]),
                    "coordinates": box.xyxy.tolist()
                })
        
        plate_text = None

        img = cv2.imread(temp_file_path)
        coords = plate_predictions[0]["coordinates"][0]
        x1, y1, x2, y2 = map(int, coords)
        cropped_plate = img[y1:y2, x1:x2]

        ocr_result = reader.readtext(cropped_plate, detail=0)
        if ocr_result:
            plate_text = ocr_result[0]
            plate_text = re.sub(r'[^A-Za-z0-9\s]+', '', plate_text)
        
        print({
            "vehicle_predictions": vehicle_predictions,
            "plate_predictions": plate_predictions,
            "plate_text": plate_text
        })
        return JSONResponse(content={
            "vehicle_predictions": vehicle_predictions,
            "plate_predictions": plate_predictions,
            "plate_text": plate_text
        })
    finally:
        # Ensure the temporary file is removed even if an error occurs
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
@app.get("/")
async def root():
    return {"message": "Welcome to the YOLOv8 API. Use /predict to upload an image for object detection."}