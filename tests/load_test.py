"""
Load testing with Locust
"""
from locust import HttpUser, task, between
from PIL import Image
import io


class APIUser(HttpUser):
    """Simulated API user"""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks
    
    def on_start(self):
        """Setup - create test image"""
        # Create a sample image
        img = Image.new('RGB', (640, 480), color='blue')
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        self.test_image = img_byte_arr.getvalue()
    
    @task(3)
    def predict(self):
        """Test prediction endpoint (most common)"""
        files = {"file": ("test.jpg", self.test_image, "image/jpeg")}
        self.client.post("/predict", files=files)
    
    @task(2)
    def detect_only(self):
        """Test detection-only endpoint"""
        files = {"file": ("test.jpg", self.test_image, "image/jpeg")}
        self.client.post("/detect-only", files=files)
    
    @task(1)
    def health_check(self):
        """Test health check endpoint"""
        self.client.get("/health")
    
    @task(1)
    def model_info(self):
        """Test model info endpoint"""
        self.client.get("/model-info")


# Run with: locust -f tests/load_test.py --host=http://localhost:8000
