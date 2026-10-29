"""TrustBridge backend (FastAPI).  Run: uvicorn main:app --reload"""
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE = Path(__file__).parent
bundle = joblib.load(BASE / "model.joblib")   # run train.py first
MODEL, SCALER, FEATURES, METRICS = bundle["model"], bundle["scaler"], bundle["features"], bundle["metrics"]

# Each persona re-weights how much each signal counts (1.0 = neutral).
PERSONAS = {
    "balanced": {"name": "Balanced", "description": "Treats every signal equally.",
                 "weights": {}},
    "security": {"name": "Security-focused", "description": "Cares most about verification and reliability.",
                 "weights": {"verification": 1.8, "reliability": 1.5, "community": 0.7}},
    "community": {"name": "Community-focused", "description": "Cares most about community feedback and transparency.",
                  "weights": {"community": 1.8, "transparency": 1.4, "verification": 0.8}},
    "performance": {"name": "Performance-focused", "description": "Cares most about responsiveness and reliability.",
                    "weights": {"responsiveness": 1.8, "reliability": 1.4, "community": 0.8}},
}

app = FastAPI(title="TrustBridge", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Signals(BaseModel):
    reliability: float = Field(ge=0, le=100)
    transparency: float = Field(ge=0, le=100)
    responsiveness: float = Field(ge=0, le=100)
    community: float = Field(ge=0, le=100)
    verification: float = Field(ge=0, le=100)


def evaluate(signals: Signals, persona_key: str = "balanced") -> dict:
    persona = PERSONAS[persona_key]
    values = signals.model_dump()
    x = np.array([[values[f] for f in FEATURES]])
    z = SCALER.transform(x)[0]                                   # feature scaling
    w = np.array([persona["weights"].get(f, 1.0) for f in FEATURES])
    contrib = MODEL.coef_[0] * z * w                             # per-feature push on the logit
    logit = MODEL.intercept_[0] + contrib.sum()
    prob = float(1 / (1 + np.exp(-logit)))                       # sigmoid

    ranked = sorted(zip(FEATURES, contrib), key=lambda t: t[1], reverse=True)
    explanation = [{"feature": f, "contribution": round(float(c), 3)} for f, c in ranked]
    positives = [e["feature"] for e in explanation if e["contribution"] > 0][:3]
    weakest = [e["feature"] for e in explanation if e["contribution"] < 0][-2:]

    if prob >= 0.75:
        label, rec = "High trust", "Safe to proceed."
    elif prob >= 0.45:
        label, rec = "Moderate trust", "Proceed with some checks."
    else:
        label, rec = "Low trust", "Be cautious and verify independently."
    if weakest:
        rec += " Weakest areas for this persona: " + ", ".join(weakest) + "."

    return {
        "trust_score": round(prob * 100, 1),
        "label": label,
        "persona": {"key": persona_key, "name": persona["name"]},
        "positive_factors": positives,
        "needs_improvement": weakest,
        "explanation": explanation,
        "recommendation": rec,
    }


@app.get("/")
def root():
    return {"status": "ok", "service": "TrustBridge"}


@app.get("/personas")
def personas():
    return {k: {"name": v["name"], "description": v["description"]} for k, v in PERSONAS.items()}


@app.post("/score")
def score(signals: Signals):
    return evaluate(signals, "balanced")


@app.post("/score/persona/{key}")
def score_persona(key: str, signals: Signals):
    if key not in PERSONAS:
        raise HTTPException(404, f"Unknown persona '{key}'. Options: {list(PERSONAS)}")
    return evaluate(signals, key)


@app.get("/model-info")
def model_info():
    return {
        "algorithm": "LogisticRegression",
        "preprocessing": "StandardScaler (z = (x - mean) / std)",
        "features": FEATURES,
        "coefficients": {f: round(float(c), 3) for f, c in zip(FEATURES, MODEL.coef_[0])},
        "intercept": round(float(MODEL.intercept_[0]), 3),
        "metrics": METRICS,
    }


# Frontend served at /app
app.mount("/app", StaticFiles(directory=BASE / "static", html=True), name="static")