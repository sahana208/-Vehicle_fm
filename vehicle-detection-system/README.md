# Vehicle Detection and License Plate Recognition System

A web-based system that detects vehicles in images using YOLOv8 and extracts license plate numbers using EasyOCR.

## Features

- **Vehicle Detection**: Detects cars, buses, trucks, and motorcycles using YOLOv8 pretrained model
- **License Plate Detection**: Identifies and crops license plate regions from detected vehicles
- **OCR Text Extraction**: Extracts license plate text using EasyOCR
- **Bounding Box Visualization**: Draws bounding boxes around detected vehicles and labels
- **Web Interface**: User-friendly React frontend for image upload and results display

## Tech Stack

### Backend
- **FastAPI**: Web framework for the API
- **YOLOv8 (Ultralytics)**: Vehicle and license plate detection
- **EasyOCR**: Optical character recognition for license plates
- **OpenCV**: Image processing
- **NumPy**: Numerical operations

### Frontend
- **React**: UI framework
- **Axios**: HTTP client for API requests
- **CSS**: Styling with gradients and modern design

## Project Structure

```
vehicle-detection-system/
├── backend/
│   ├── main.py              # FastAPI application
│   └── requirements.txt     # Python dependencies
└── frontend/
    ├── package.json         # Node dependencies
    ├── public/
    │   └── index.html       # HTML template
    └── src/
        ├── App.js           # Main React component
        ├── App.css          # Component styles
        ├── index.js         # React entry point
        └── index.css        # Global styles
```

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Node.js 16 or higher
- pip (Python package manager)
- npm or yarn (Node package manager)

### Backend Setup

1. Navigate to the backend directory:
```bash
cd vehicle-detection-system/backend
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
```

3. Activate the virtual environment:
- Windows: `venv\Scripts\activate`
- Linux/Mac: `source venv/bin/activate`

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Start the backend server:
```bash
python main.py
```

The backend will start on `http://localhost:8000`

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd vehicle-detection-system/frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the frontend development server:
```bash
npm start
```

The frontend will open on `http://localhost:3000`

## Usage

1. Open the web application at `http://localhost:3000`
2. Click "Choose an image" to upload an image containing vehicles
3. Click "Detect Vehicles" to process the image
4. View the results:
   - Processed image with bounding boxes
   - Detected vehicle types (car, bus, truck, motorcycle)
   - Extracted license plate numbers
   - Confidence scores for each detection

## API Endpoint

### POST /detect

Upload an image for vehicle detection and license plate recognition.

**Request**: Multipart form data with a file named "file"

**Response**: JSON object containing:
- `success`: Boolean indicating success
- `results`: Array of detection results with:
  - `vehicle_type`: Type of vehicle (car, bus, truck, motorcycle)
  - `confidence`: Detection confidence score (0-1)
  - `plate_number`: Extracted license plate text (or null if not detected)
  - `bbox`: Bounding box coordinates [x1, y1, x2, y2]
- `image`: Base64-encoded processed image

## Model Information

- **Vehicle Detection**: YOLOv8 Nano (yolov8n.pt) - automatically downloaded on first run
- **License Plate Detection**: YOLOv8 Nano (reused for plate detection)
- **OCR**: EasyOCR with English language support

## Notes

- The first run will download the YOLOv8 model (~6MB)
- EasyOCR will download language models on first use (~100MB)
- GPU acceleration is automatically used if CUDA is available
- License plate detection uses the vehicle region to improve accuracy

## Troubleshooting

**Issue**: "Module not found" errors
- **Solution**: Ensure all dependencies are installed in the virtual environment

**Issue**: Frontend cannot connect to backend
- **Solution**: Ensure backend is running on port 8000 and CORS is enabled

**Issue**: Slow processing
- **Solution**: Enable GPU acceleration by installing CUDA-compatible PyTorch

**Issue**: Poor OCR accuracy
- **Solution**: Ensure license plates are clearly visible and well-lit in the image
