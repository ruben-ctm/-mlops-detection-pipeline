"""
Download pre-trained models
"""
from ultralytics import YOLO
from transformers import CLIPModel, CLIPProcessor
import os


def download_yolo():
    """Download YOLOv8 models"""
    print("Downloading YOLOv8 models...")
    
    models = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt']
    
    for model_name in models:
        print(f"  - {model_name}")
        model = YOLO(model_name)
    
    print("✓ YOLOv8 models downloaded")


def download_clip():
    """Download CLIP model"""
    print("\nDownloading CLIP model...")
    
    model_name = "openai/clip-vit-base-patch32"
    print(f"  - {model_name}")
    
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    
    print("✓ CLIP model downloaded")


def main():
    print("="*60)
    print("Downloading pre-trained models...")
    print("="*60)
    
    download_yolo()
    download_clip()
    
    print("\n" + "="*60)
    print("All models downloaded successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
