import pytest
from pydantic import ValidationError
from schemas import PredictRequest, PredictResponse, TrainingStartRequest
from datetime import datetime

class TestPredictRequest:
    def test_valid_request(self):
        request = PredictRequest(user_id="u1", text="text", model_name="Default")
        assert request.model_name == "Default"

    def test_optional_model_name(self):
        request = PredictRequest(user_id="u1", text="text")
        assert request.model_name is None

class TestTrainingStartRequest:
    def test_valid_training_request(self):
        req = TrainingStartRequest(
            dataset_path="./data.csv",
            num_epochs=5,
            custom_model_name="new_model",
            source_model_path="DEFAULT"
        )
        assert req.num_epochs == 5
        assert req.custom_model_name == "new_model"

    def test_defaults(self):
        # Проверяем дефолтные значения
        req = TrainingStartRequest(dataset_path="./data.csv")
        assert req.num_epochs == 3
        assert req.learning_rate == 2e-4
        assert req.custom_model_name == "my_model"