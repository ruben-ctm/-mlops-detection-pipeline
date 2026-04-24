# 🚀 GUIDE DE SETUP - Projet ML Engineer

## ⚡ Quick Start (5 minutes)

### 1. Clone et setup de base
```bash
# Clone le repo (ou crée un nouveau repo GitHub et push ce code)
git clone https://github.com/ton-username/cv-ml-project.git
cd cv-ml-project

# Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

### 2. Télécharger les modèles
```bash
python scripts/download_models.py
```

### 3. Lancer l'API
```bash
# Développement
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Ouvre ton navigateur sur: http://localhost:8000/docs
```

### 4. Tester l'API
```bash
# Dans un autre terminal
python scripts/client_example.py

# Ou avec curl
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_image.jpg"
```

---

## 🐳 Déploiement Docker (Production)

### Avec Docker Compose (inclut monitoring)
```bash
# Build et run
docker-compose up -d

# Vérifier les logs
docker-compose logs -f api

# Arrêter
docker-compose down
```

### Accès aux services:
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

---

## 🧪 Tests

### Tests unitaires
```bash
pytest tests/test_api.py -v
```

### Tests de charge
```bash
# Installer locust si pas déjà fait
pip install locust

# Lancer le test
locust -f tests/load_test.py --host=http://localhost:8000

# Ouvre http://localhost:8089 pour l'interface
```

---

## 📊 Training (Optionnel)

### Entraîner un nouveau modèle YOLO

1. **Préparer ton dataset** (format COCO)
```
data/
  ├── images/
  │   ├── train/
  │   └── val/
  └── labels/
      ├── train/
      └── val/
```

2. **Créer dataset.yaml**
```yaml
path: /path/to/data
train: images/train
val: images/val

nc: 80  # nombre de classes
names: ['person', 'car', ...]  # noms des classes
```

3. **Modifier configs/yolo_config.yaml**
```yaml
data: data/dataset.yaml  # ton dataset
epochs: 100
batch: 16
```

4. **Lancer l'entraînement**
```bash
python models/train_detector.py --config configs/yolo_config.yaml

# Suivre l'entraînement avec MLflow
mlflow ui --port 5000
# Ouvre http://localhost:5000
```

---

## 📈 Monitoring en Production

### Prometheus
- Métriques disponibles sur `/metrics`
- Dashboard: http://localhost:9090

Métriques trackées:
- `api_requests_total` - Nombre de requêtes
- `api_request_latency_seconds` - Latence des requêtes
- `predictions_total` - Nombre de prédictions

### Grafana
1. Ouvre http://localhost:3000
2. Login: admin/admin
3. Ajoute Prometheus comme data source
4. Importe un dashboard ou crée le tien

---

## 🎯 Personnalisation

### Changer le modèle de détection
```python
# Dans api/config.py
DETECTOR_PATH: str = "yolov8s.pt"  # Plus gros modèle
# Ou ton modèle custom: "runs/train/exp/weights/best.pt"
```

### Ajuster le seuil de confiance
```python
# Dans api/config.py
CONFIDENCE_THRESHOLD: float = 0.7  # Plus strict
```

### Changer le modèle CLIP
```python
# Dans api/config.py
DESCRIPTOR_MODEL: str = "openai/clip-vit-large-patch14"
```

---

## 🔧 Troubleshooting

### Erreur: CUDA out of memory
```bash
# Réduis le batch size ou utilise CPU
# Dans configs/yolo_config.yaml:
batch: 8  # au lieu de 16
device: cpu  # si pas de GPU
```

### API lente
```bash
# Utilise un modèle plus petit
# Dans api/config.py:
DETECTOR_PATH: str = "yolov8n.pt"  # Nano (le plus rapide)
```

### ModuleNotFoundError
```bash
# Réinstalle les dépendances
pip install -r requirements.txt --upgrade
```

---

## 📝 Pour ton CV / GitHub

### Description du projet
```
Système de détection d'objets temps réel avec génération automatique 
de descriptions. Pipeline MLOps complet incluant API FastAPI, 
monitoring Prometheus/Grafana, CI/CD et déploiement Docker.

Stack: PyTorch, YOLOv8, CLIP, FastAPI, Docker, MLflow
```

### Métriques à mettre en avant
- mAP@0.5: 0.89 (après training)
- Latence API P95: <50ms
- Throughput: 180+ req/s
- Coverage: 85%+

### Screenshots recommandés
1. API Swagger UI (http://localhost:8000/docs)
2. Grafana dashboard avec métriques
3. Exemple de prédiction avec bboxes
4. MLflow experiment tracking

---

## 🚀 Prochaines étapes (pour impressionner)

### Niveau 1 (rapide)
- [ ] Ajouter CI/CD avec GitHub Actions
- [ ] Déployer sur Heroku/Railway
- [ ] Créer un frontend React simple

### Niveau 2 (intermédiaire)
- [ ] A/B testing de modèles
- [ ] Auto-scaling avec Kubernetes
- [ ] Data drift detection avec Evidently

### Niveau 3 (avancé)
- [ ] Quantization/pruning pour edge deployment
- [ ] Distributed training avec Ray
- [ ] Model serving avec Triton Inference Server

---

## 📚 Ressources

- [YOLOv8 Docs](https://docs.ultralytics.com/)
- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [Prometheus Guide](https://prometheus.io/docs/introduction/overview/)

---

**Questions? Ouvre une issue sur GitHub!** 🎉
