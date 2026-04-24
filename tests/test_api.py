"""
Unit tests for API endpoints
"""
import pytest
from fastapi.testclient import TestClient
from PIL import Image
import io
import numpy as np


@pytest.fixture
def client():
    """Create test client"""
    from api.main import app
    return TestClient(app)


@pytest.fixture
def sample_image():
    """Create a sample test image"""
    # Create a simple RGB image
    img = Image.new('RGB', (640, 480), color='red')
    
    # Convert to bytes
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    
    return img_byte_arr


def test_root_endpoint(client):
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data


def test_health_check(client):
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "models_loaded" in data


def test_model_info(client):
    """Test model info endpoint"""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "detector" in data
    assert "descriptor" in data


def test_predict_endpoint(client, sample_image):
    """Test prediction endpoint"""
    files = {"file": ("test.jpg", sample_image, "image/jpeg")}
    response = client.post("/predict", files=files)
    
    assert response.status_code == 200
    data = response.json()
    
    assert "success" in data
    assert "detections" in data
    assert "inference_time_ms" in data
    assert isinstance(data["detections"], list)


def test_detect_only_endpoint(client, sample_image):
    """Test detection-only endpoint"""
    files = {"file": ("test.jpg", sample_image, "image/jpeg")}
    response = client.post("/detect-only", files=files)
    
    assert response.status_code == 200
    data = response.json()
    
    assert "success" in data
    assert "detections" in data
    assert "inference_time_ms" in data


def test_invalid_file_format(client):
    """Test with invalid file format"""
    files = {"file": ("test.txt", b"not an image", "text/plain")}
    response = client.post("/predict", files=files)
    
    assert response.status_code == 500


def test_metrics_endpoint(client):
    """Test Prometheus metrics endpoint"""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]


@pytest.mark.parametrize("endpoint", ["/predict", "/detect-only"])
def test_response_format(client, sample_image, endpoint):
    """Test response format for different endpoints"""
    files = {"file": ("test.jpg", sample_image, "image/jpeg")}
    response = client.post(endpoint, files=files)
    
    assert response.status_code == 200
    data = response.json()
    
    # Check required fields
    assert "success" in data
    assert data["success"] is True
    assert "detections" in data
    assert isinstance(data["detections"], list)


def test_concurrent_requests(client, sample_image):
    """Test handling concurrent requests"""
    import concurrent.futures
    
    def make_request():
        files = {"file": ("test.jpg", sample_image, "image/jpeg")}
        return client.post("/predict", files=files)
    
    # Make 10 concurrent requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_request) for _ in range(10)]
        responses = [f.result() for f in futures]
    
    # All should succeed
    assert all(r.status_code == 200 for r in responses)
