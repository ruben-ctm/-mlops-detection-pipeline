"""
Example Python client for the API
"""
import requests
from pathlib import Path
import json


class DetectionClient:
    """Client for Object Detection API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def health_check(self):
        """Check API health"""
        response = requests.get(f"{self.base_url}/health")
        return response.json()
    
    def predict(self, image_path: str):
        """
        Get predictions with descriptions
        
        Args:
            image_path: Path to image file
            
        Returns:
            API response with detections
        """
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                f"{self.base_url}/predict",
                files=files
            )
        
        return response.json()
    
    def detect_only(self, image_path: str):
        """
        Get detections only (faster)
        
        Args:
            image_path: Path to image file
            
        Returns:
            API response with detections
        """
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                f"{self.base_url}/detect-only",
                files=files
            )
        
        return response.json()
    
    def model_info(self):
        """Get model information"""
        response = requests.get(f"{self.base_url}/model-info")
        return response.json()


def main():
    """Example usage"""
    
    # Initialize client
    client = DetectionClient()
    
    # Check health
    print("Checking API health...")
    health = client.health_check()
    print(f"Status: {health['status']}")
    print(f"Models loaded: {health['models_loaded']}")
    print()
    
    # Get model info
    print("Model information:")
    info = client.model_info()
    print(json.dumps(info, indent=2))
    print()
    
    # Example prediction
    image_path = "test_image.jpg"
    
    if Path(image_path).exists():
        print(f"Running prediction on {image_path}...")
        result = client.predict(image_path)
        
        print(f"Success: {result['success']}")
        print(f"Inference time: {result['inference_time_ms']}ms")
        print(f"Detections: {len(result['detections'])}")
        print()
        
        for i, det in enumerate(result['detections'], 1):
            print(f"Detection {i}:")
            print(f"  Class: {det['class_name']}")
            print(f"  Confidence: {det['confidence']:.2%}")
            print(f"  BBox: {det['bbox']}")
            print(f"  Description: {det['description']}")
            print()
    else:
        print(f"Image {image_path} not found!")
        print("Place a test image in the current directory to try predictions.")


if __name__ == "__main__":
    main()
