"""
Training script for YOLOv8 object detector
"""
import argparse
from ultralytics import YOLO
import yaml
import mlflow
import torch
from pathlib import Path


def train(config_path: str):
    """
    Train YOLOv8 model
    
    Args:
        config_path: Path to config YAML file
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize MLflow
    mlflow.set_tracking_uri(config.get('mlflow_uri', 'mlruns'))
    mlflow.set_experiment(config.get('experiment_name', 'yolo-training'))
    
    with mlflow.start_run():
        # Log config
        mlflow.log_params(config)
        
        # Initialize model
        model = YOLO(config['model'])  # yolov8n.pt, yolov8s.pt, etc.
        
        # Train
        results = model.train(
            data=config['data'],
            epochs=config['epochs'],
            imgsz=config['imgsz'],
            batch=config['batch'],
            device=config.get('device', 0),
            workers=config.get('workers', 8),
            patience=config.get('patience', 50),
            save=True,
            project=config.get('project', 'runs/train'),
            name=config.get('name', 'exp'),
            exist_ok=True,
            pretrained=True,
            optimizer=config.get('optimizer', 'SGD'),
            lr0=config.get('lr0', 0.01),
            lrf=config.get('lrf', 0.01),
            momentum=config.get('momentum', 0.937),
            weight_decay=config.get('weight_decay', 0.0005),
            warmup_epochs=config.get('warmup_epochs', 3.0),
            warmup_momentum=config.get('warmup_momentum', 0.8),
            box=config.get('box', 7.5),
            cls=config.get('cls', 0.5),
            dfl=config.get('dfl', 1.5),
            hsv_h=config.get('hsv_h', 0.015),
            hsv_s=config.get('hsv_s', 0.7),
            hsv_v=config.get('hsv_v', 0.4),
            degrees=config.get('degrees', 0.0),
            translate=config.get('translate', 0.1),
            scale=config.get('scale', 0.5),
            shear=config.get('shear', 0.0),
            perspective=config.get('perspective', 0.0),
            flipud=config.get('flipud', 0.0),
            fliplr=config.get('fliplr', 0.5),
            mosaic=config.get('mosaic', 1.0),
            mixup=config.get('mixup', 0.0),
            copy_paste=config.get('copy_paste', 0.0)
        )
        
        # Log metrics
        metrics = results.results_dict
        mlflow.log_metrics({
            'mAP50': metrics.get('metrics/mAP50(B)', 0),
            'mAP50-95': metrics.get('metrics/mAP50-95(B)', 0),
            'precision': metrics.get('metrics/precision(B)', 0),
            'recall': metrics.get('metrics/recall(B)', 0),
        })
        
        # Validate
        val_results = model.val()
        mlflow.log_metrics({
            'val_mAP50': val_results.box.map50,
            'val_mAP50-95': val_results.box.map,
        })
        
        # Log model
        best_model_path = Path(results.save_dir) / 'weights' / 'best.pt'
        mlflow.log_artifact(str(best_model_path))
        
        print(f"\n{'='*50}")
        print(f"Training completed!")
        print(f"Best model saved to: {best_model_path}")
        print(f"mAP@0.5: {metrics.get('metrics/mAP50(B)', 0):.4f}")
        print(f"mAP@0.5:0.95: {metrics.get('metrics/mAP50-95(B)', 0):.4f}")
        print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 detector')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/yolo_config.yaml',
        help='Path to config file'
    )
    
    args = parser.parse_args()
    train(args.config)


if __name__ == '__main__':
    main()
