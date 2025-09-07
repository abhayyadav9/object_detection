from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Real-time Object Detection Backend!"}

# Placeholder for other API tests
# def test_status_endpoint():
#     response = client.get("/status")
#     assert response.status_code == 200
#     assert "model_loaded" in response.json()
