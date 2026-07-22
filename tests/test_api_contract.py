"""Inference contract tests.

Production concept: the API's request/response shape is a CONTRACT with every
client. These tests pin it down so a refactor can't silently break callers.
They also encode the latency budget and input-validation behavior.

Run (API must be up):  pytest tests/test_api_contract.py -v
"""
import numpy as np
import pytest
import requests

API = "http://localhost:8000"
LATENCY_BUDGET_MS = 100  # generous for CPU ONNX on a 28x28 image


def valid_payload():
    rng = np.random.default_rng(1)
    return {"pixels": rng.random(784).tolist()}


def test_health():
    r = requests.get(f"{API}/health", timeout=5)
    assert r.status_code == 200
    assert r.json()["status"] in ("ready", "no_model")


def test_predict_response_shape():
    r = requests.post(f"{API}/predict", json=valid_payload(), timeout=5)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"digit", "confidence", "latency_ms"}
    assert 0 <= body["digit"] <= 9
    assert 0.0 <= body["confidence"] <= 1.0


def test_latency_budget():
    r = requests.post(f"{API}/predict", json=valid_payload(), timeout=5)
    assert r.json()["latency_ms"] < LATENCY_BUDGET_MS


@pytest.mark.parametrize("bad_pixels", [
    [0.5] * 100,            # wrong length
    [0.5] * 785,            # wrong length
    [2.0] * 784,            # out of range
    [-0.1] * 784,           # out of range
])
def test_rejects_invalid_input(bad_pixels):
    r = requests.post(f"{API}/predict", json={"pixels": bad_pixels}, timeout=5)
    assert r.status_code == 422  # validation error, NOT a 500 or a garbage prediction
