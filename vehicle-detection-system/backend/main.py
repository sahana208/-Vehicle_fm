from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import cv2
import numpy as np
import base64
import easyocr
import torch
import re

from ultralytics import YOLO

app = FastAPI()

# =========================
# CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# LOAD MODELS
# =========================

# Vehicle Detection Model
vehicle_model = YOLO("yolov8n.pt")

# License Plate Detection Model
# Download a license plate model and place it in project folder
# Example name: license_plate_detector.pt
plate_model = YOLO("license_plate_detector.pt")

# OCR Reader
reader = easyocr.Reader(
    ['en'],
    gpu=torch.cuda.is_available()
)

# =========================
# VEHICLE CLASSES
# =========================

VEHICLE_CLASSES = {
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck'
}

# =========================
# VEHICLE DETECTION
# =========================

def detect_vehicles(image):

    detections = []

    results = vehicle_model(
        image,
        conf=0.5,
        iou=0.45
    )

    for result in results:

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            if class_id not in VEHICLE_CLASSES:
                continue

            if confidence < 0.5:
                continue

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "vehicle_type": VEHICLE_CLASSES[class_id],
                "confidence": round(confidence, 2)
            })

    return detections

# =========================
# LICENSE PLATE DETECTION
# =========================

def detect_license_plates(image):

    plates = []

    results = plate_model(
        image,
        conf=0.5,
        iou=0.45
    )

    for result in results:

        for box in result.boxes:

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            plates.append([x1, y1, x2, y2])

    return plates

# =========================
# OCR EXTRACTION
# =========================

def extract_plate_text(image, plate_bbox):

    try:

        x1, y1, x2, y2 = plate_bbox

        h, w = image.shape[:2]

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        plate_region = image[y1:y2, x1:x2]

        if plate_region.size == 0:
            return None

        # =========================
        # PREPROCESSING
        # =========================

        gray = cv2.cvtColor(
            plate_region,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.bilateralFilter(
            gray,
            11,
            17,
            17
        )

        gray = cv2.resize(
            gray,
            None,
            fx=3,
            fy=3,
            interpolation=cv2.INTER_CUBIC
        )

        _, thresh = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # =========================
        # OCR
        # =========================

        results = reader.readtext(
            thresh,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        )

        if not results:
            return None

        # Highest confidence text
        best_result = max(results, key=lambda x: x[2])

        text = best_result[1]

        # Clean text
        text = text.upper()
        text = text.replace(" ", "")

        # Remove symbols
        text = re.sub(r'[^A-Z0-9]', '', text)

        # Indian number plate validation
        pattern = r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$'

        if re.match(pattern, text):
            return text

        return text

    except Exception as e:
        print("OCR Error:", e)
        return None

# =========================
# DRAW RESULTS
# =========================

def draw_results(
    image,
    vehicles,
    plates,
    plate_texts
):

    img = image.copy()

    # Draw vehicles
    for vehicle in vehicles:

        x1, y1, x2, y2 = vehicle["bbox"]

        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        label = f"{vehicle['vehicle_type']} {vehicle['confidence']}"

        cv2.putText(
            img,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # Draw plates
    for i, plate in enumerate(plates):

        x1, y1, x2, y2 = plate

        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )

        if i < len(plate_texts):

            text = plate_texts[i]

            if text:

                cv2.putText(
                    img,
                    text,
                    (x1, y2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 0),
                    2
                )

    return img

# =========================
# API ENDPOINT
# =========================

@app.post("/detect")

async def detect(file: UploadFile = File(...)):

    try:

        contents = await file.read()

        nparr = np.frombuffer(
            contents,
            np.uint8
        )

        image = cv2.imdecode(
            nparr,
            cv2.IMREAD_COLOR
        )

        if image is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid image"
            )

        # =========================
        # VEHICLE DETECTION
        # =========================

        vehicles = detect_vehicles(image)

        # =========================
        # LICENSE PLATE DETECTION
        # =========================

        plates = detect_license_plates(image)

        # =========================
        # OCR
        # =========================

        plate_texts = []

        for plate in plates:

            text = extract_plate_text(
                image,
                plate
            )

            plate_texts.append(text)

        # =========================
        # DRAW OUTPUT
        # =========================

        result_img = draw_results(
            image,
            vehicles,
            plates,
            plate_texts
        )

        # =========================
        # IMAGE TO BASE64
        # =========================

        _, buffer = cv2.imencode(
            ".jpg",
            result_img
        )

        img_base64 = base64.b64encode(
            buffer
        ).decode("utf-8")

        # =========================
        # RESPONSE
        # =========================

        response_results = []

        for vehicle in vehicles:

            response_results.append({
                "vehicle_type": vehicle["vehicle_type"],
                "confidence": vehicle["confidence"],
                "bbox": vehicle["bbox"]
            })

        return JSONResponse({

            "success": True,

            "vehicles": response_results,

            "plates": plate_texts,

            "image": f"data:image/jpeg;base64,{img_base64}"

        })

    except Exception as e:

        print("ERROR:", e)

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# =========================
# ROOT
# =========================

@app.get("/")

async def root():

    return {
        "message": "Vehicle Detection + License Plate Recognition API"
    }

# =========================
# MAIN
# =========================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )