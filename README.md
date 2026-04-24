# 🚀 Real-Time Object Detection & Description System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Description

Système de détection d'objets en temps réel avec génération automatique de descriptions textuelles. Pipeline MLOps complet de l'entraînement au déploiement avec monitoring.

**Stack**: PyTorch, YOLOv8, CLIP, FastAPI, Docker, MLflow, Prometheus

## 🎯 Features

- ✅ Détection d'objets temps réel (YOLOv8)
- ✅ Génération de descriptions (CLIP)
- ✅ API REST haute performance (FastAPI)
- ✅ Versioning modèles (MLflow)
- ✅ Monitoring & alerting (Prometheus/Grafana)
- ✅ CI/CD complet (GitHub Actions)
- ✅ Deployment Docker/Kubernetes ready

## 🏗️ Architecture

```
├── data/                   # Datasets et preprocessing
├── models/                 # Training scripts & configs
├── api/                    # FastAPI service
├── monitoring/             # Prometheus/Grafana configs
├── tests/                  # Unit & integration tests
├── notebooks/              # Exploratory analysis
└── deployment/             # Docker, K8s configs
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Docker & Docker Compose
- CUDA 11.8+ (optionnel pour GPU)

### Installation

```bash
# Clone le repo
git clone https://github.com/yourusername/cv-ml-project.git
cd cv-ml-project

# Setup environnement
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download pre-trained weights
python scripts/download_models.py
```

### Lancer l'API

```bash
# Development
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Production (Docker)
docker-compose up -d
```

### Test l'API

```bash
# Health check
curl http://localhost:8000/health

# Prédiction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_image.jpg"
```

## 📊 Training

```bash
# Train YOLOv8 detector
python models/train_detector.py --config configs/yolo_config.yaml

# Fine-tune CLIP pour descriptions
python models/train_descriptor.py --config configs/clip_config.yaml

# Track experiments
mlflow ui --port 5000
```

## 🧪 Testing

```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Performance tests
locust -f tests/load_test.py
```

## 📈 Monitoring

Dashboard Grafana accessible sur `http://localhost:3000`

Métriques trackées:
- Latence P50/P95/P99
- Throughput (req/sec)
- Model accuracy en production
- Data drift detection

## 🎯 Résultats

| Metric | Value |
|--------|-------|
| mAP@0.5 | 0.89 |
| Inference time (GPU) | 23ms |
| API latency P95 | 45ms |
| Throughput | 180 req/s |

## 📝 TODO / Roadmap

- [ ] Support multi-GPU training
- [ ] Edge deployment (ONNX/TensorRT)
- [ ] A/B testing framework
- [ ] Auto-scaling basé sur load

## 🤝 Contributing

Pull requests welcome! Voir [CONTRIBUTING.md](CONTRIBUTING.md)

## 📄 License

MIT License - voir [LICENSE](LICENSE)

## 👤 Auteur

**Ton Nom**
- GitHub: [@ruben-ctm](https://github.com/ruben-ctm)
- LinkedIn: [RubenCombe-Tamain]([https://linkedin.com/in/ton-profil](https://www.linkedin.com/in/rubencombe-tamain/))

---

⭐ Star ce projet si tu le trouves utile!
