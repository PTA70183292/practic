import pytest
from fastapi import status
import os

class TestHealthCheck:
    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {"status": "ok"}

class TestPredictEndpoint:
    def test_predict_basic(self, client):
        # Тест без указания модели
        payload = {"user_id": "test_u", "text": "Хороший товар"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "score" in data
        assert "label" in data

    def test_predict_with_model_name(self, client):
        # Тест с указанием модели
        payload = {
            "user_id": "test_u", 
            "text": "Плохой товар", 
            "model_name": "Default"
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

class TestTrainingEndpoints:
    def test_get_models_list(self, client):
        response = client.get("/training/models-list")
        assert response.status_code == 200
        assert "models" in response.json()
        assert isinstance(response.json()["models"], list)

    def test_start_training_validation(self, client):
        # Пытаемся запустить обучение с несуществующим файлом
        payload = {
            "dataset_path": "./non_existent.csv",
            "num_epochs": 1,
            "custom_model_name": "test_run"
        }
        response = client.post("/training/start", json=payload)
        # Должен вернуть 404, так как файл не найден
        assert response.status_code == 404

class TestHistory:
    def test_get_user_history(self, client):
        user_id = "history_user"
        client.post("/predict", json={"user_id": user_id, "text": "1"})
        client.post("/predict", json={"user_id": user_id, "text": "2"})
        
        response = client.get(f"/predictions/user/{user_id}")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["user_id"] == user_id