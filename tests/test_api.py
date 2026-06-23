from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200

def test_predict_format():
    response = client.post("/predict", json={
        "surface": 65.0,
        "nb_pieces": 3,
        "code_departement": 75
    })
    assert response.status_code == 200
    assert "prix_estime" in response.json()