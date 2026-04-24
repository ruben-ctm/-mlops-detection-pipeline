"""
FastAPI service pour détection d'objets et génération de descriptions
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import torch
from PIL import Image
import io
import time
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response
import logging

from .models import ObjectDetector, DescriptionGenerator
from .config import Settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load settings
settings = Settings()

# Initialize FastAPI
app = FastAPI(
    title="Object Detection & Description API",
    description="Real-time object detection with automatic description generation",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['endpoint', 'status'])
REQUEST_LATENCY = Histogram('api_request_latency_seconds', 'Request latency', ['endpoint'])
PREDICTION_COUNT = Counter('predictions_total', 'Total predictions made')

# Load models at startup
detector = None
descriptor = None

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    global detector, descriptor
    logger.info("Loading models...")
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Load YOLO detector
        detector = ObjectDetector(
            model_path=settings.DETECTOR_PATH,
            device=device,
            conf_threshold=settings.CONFIDENCE_THRESHOLD
        )
        
        # Load CLIP descriptor
        descriptor = DescriptionGenerator(
            model_name=settings.DESCRIPTOR_MODEL,
            device=device
        )
        
        logger.info("Models loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise


class DetectionResult(BaseModel):
    """Detection result schema"""
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    description: Optional[str] = None


class PredictionResponse(BaseModel):
    """API response schema"""
    success: bool
    detections: List[DetectionResult]
    inference_time_ms: float
    image_size: List[int]


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Object Detection & Description API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    models_loaded = detector is not None and descriptor is not None
    
    return {
        "status": "healthy" if models_loaded else "unhealthy",
        "models_loaded": models_loaded,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cuda_available": torch.cuda.is_available()
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Main prediction endpoint
    
    Args:
        file: Image file (JPEG, PNG)
        
    Returns:
        DetectionResponse with detected objects and descriptions
    """
    start_time = time.time()
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_size = list(image.size)
        
        # Detect objects
        detections = detector.predict(image)
        
        # Generate descriptions for each detection
        results = []
        for det in detections:
            # Crop detected region
            x1, y1, x2, y2 = det["bbox"]
            cropped = image.crop((x1, y1, x2, y2))
            
            # Generate description
            description = descriptor.generate_description(cropped)
            
            results.append(DetectionResult(
                class_name=det["class_name"],
                confidence=det["confidence"],
                bbox=det["bbox"],
                description=description
            ))
        
        inference_time = (time.time() - start_time) * 1000
        
        # Update metrics
        PREDICTION_COUNT.inc()
        REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)
        REQUEST_COUNT.labels(endpoint="/predict", status="success").inc()
        
        return PredictionResponse(
            success=True,
            detections=results,
            inference_time_ms=round(inference_time, 2),
            image_size=image_size
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        REQUEST_COUNT.labels(endpoint="/predict", status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect-only")
async def detect_only(file: UploadFile = File(...)):
    """
    Detection only endpoint (faster, no descriptions)
    """
    start_time = time.time()
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        detections = detector.predict(image)
        inference_time = (time.time() - start_time) * 1000
        
        return {
            "success": True,
            "detections": detections,
            "inference_time_ms": round(inference_time, 2)
        }
        
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type="text/plain")


@app.get("/model-info")
async def model_info():
    """Get information about loaded models"""
    return {
        "detector": {
            "type": "YOLOv8",
            "path": settings.DETECTOR_PATH,
            "confidence_threshold": settings.CONFIDENCE_THRESHOLD
        },
        "descriptor": {
            "type": "CLIP",
            "model": settings.DESCRIPTOR_MODEL
        },
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
