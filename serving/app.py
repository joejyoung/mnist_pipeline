"""Inference service: the model as a product.

Production concepts demonstrated:
  * Stable artifact (ONNX) loaded at startup — no training code in this container.
  * /health endpoint — load balancers and orchestrators use this to route traffic.
  * Input validation at the boundary — reject malformed input before it hits the model.
  * Every prediction logged (label, confidence, input stats, latency) — this event
    stream is the raw material for drift monitoring and the human-review feedback loop.
  * Latency budget awareness — latency_ms is recorded per request.

Contract:  POST /predict  {"pixels": [784 floats in 0..1]}
           → {"digit": 7, "confidence": 0.98, "latency_ms": 1.9}
"""
import os
import time

import numpy as np
import onnxruntime as ort
import psycopg2
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/model.onnx")
DATABASE_URL = os.environ.get("DATABASE_URL")

# Same normalization the model was trained with — a classic source of
# training/serving skew when it silently diverges. Keep it in ONE place per side
# and covered by contract tests.
MEAN, STD = 0.1307, 0.3081

app = FastAPI(title="MNIST Inference Service")
session = None


class PredictRequest(BaseModel):
    pixels: list[float]

    @field_validator("pixels")
    @classmethod
    def check_shape_and_range(cls, v):
        if len(v) != 784:
            raise ValueError(f"expected 784 pixels, got {len(v)}")
        if min(v) < 0.0 or max(v) > 1.0:
            raise ValueError("pixel values must be in [0, 1]")
        return v


@app.on_event("startup")
def load_model():
    global session
    if os.path.exists(MODEL_PATH):
        session = ort.InferenceSession(MODEL_PATH)
        print(f"Loaded model: {MODEL_PATH}")
    else:
        print(f"WARNING: no model at {MODEL_PATH} — /predict will 503 until one exists")


@app.get("/health")
def health():
    """Orchestrator probe: 'ready' only when a model is actually loaded."""
    return {"status": "ready" if session else "no_model", "model_path": MODEL_PATH}


def log_prediction(digit, confidence, mean_px, std_px, latency_ms):
    """Fire-and-forget event log. Production: async producer to Kafka."""
    try:
        with psycopg2.connect(DATABASE_URL) as conn, conn.cursor() as cur:
            cur.execute(
                """INSERT INTO predictions
                   (predicted, confidence, mean_pixel, std_pixel, latency_ms, model_path)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (digit, confidence, mean_px, std_px, latency_ms, MODEL_PATH),
            )
    except Exception as e:  # never fail a request because logging hiccuped
        print(f"prediction log failed: {e}")


@app.post("/predict")
def predict(req: PredictRequest):
    if session is None:
        raise HTTPException(503, "model not loaded")

    t0 = time.perf_counter()
    raw = np.array(req.pixels, dtype=np.float32)
    x = ((raw - MEAN) / STD).reshape(1, 1, 28, 28)

    logits = session.run(["logits"], {"image": x})[0][0]
    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum()
    digit = int(probs.argmax())
    confidence = float(probs[digit])
    latency_ms = (time.perf_counter() - t0) * 1000

    log_prediction(digit, confidence, float(raw.mean()), float(raw.std()), latency_ms)
    return {"digit": digit, "confidence": round(confidence, 4),
            "latency_ms": round(latency_ms, 2)}
