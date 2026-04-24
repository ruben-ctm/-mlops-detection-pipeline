"""
Object detection module using YOLOv8
"""
from ultralytics import YOLO
import torch
from PIL import Image
import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class ObjectDetector:
    """YOLOv8 object detector"""
    
    def __init__(
        self, 
        model_path: str = "yolov8n.pt",
        device: str = "cuda",
        conf_threshold: float = 0.5
    ):
        """
        Initialize detector
        
        Args:
            model_path: Path to YOLO weights
            device: 'cuda' or 'cpu'
            conf_threshold: Confidence threshold for detections
        """
        self.device = device
        self.conf_threshold = conf_threshold
        
        try:
            self.model = YOLO(model_path)
            self.model.to(device)
            logger.info(f"Loaded YOLOv8 model from {model_path}")
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
            raise
    
    def predict(self, image: Image.Image) -> List[Dict]:
        """
        Run detection on image
        
        Args:
            image: PIL Image
            
        Returns:
            List of detections with format:
            {
                'class_name': str,
                'confidence': float,
                'bbox': [x1, y1, x2, y2]
            }
        """
        try:
            # Run inference
            results = self.model(
                image,
                conf=self.conf_threshold,
                verbose=False
            )[0]
            
            detections = []
            
            # Parse results
            if results.boxes is not None:
                boxes = results.boxes.cpu().numpy()
                
                for box in boxes:
                    # Get bbox coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    # Get confidence and class
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = results.names[class_id]
                    
                    detections.append({
                        'class_name': class_name,
                        'confidence': round(confidence, 3),
                        'bbox': [
                            round(x1, 1),
                            round(y1, 1),
                            round(x2, 1),
                            round(y2, 1)
                        ]
                    })
            
            return detections
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            raise
    
    def predict_batch(self, images: List[Image.Image]) -> List[List[Dict]]:
        """
        Batch prediction
        
        Args:
            images: List of PIL Images
            
        Returns:
            List of detection lists
        """
        try:
            results = self.model(
                images,
                conf=self.conf_threshold,
                verbose=False
            )
            
            all_detections = []
            for result in results:
                detections = []
                if result.boxes is not None:
                    boxes = result.boxes.cpu().numpy()
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = result.names[class_id]
                        
                        detections.append({
                            'class_name': class_name,
                            'confidence': round(confidence, 3),
                            'bbox': [round(x1, 1), round(y1, 1), 
                                   round(x2, 1), round(y2, 1)]
                        })
                all_detections.append(detections)
            
            return all_detections
            
        except Exception as e:
            logger.error(f"Batch detection error: {e}")
            raise
