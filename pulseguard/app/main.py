import json
import math
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE = Path(__file__).resolve().parent.parent
pipe = joblib.load(BASE / "model" / "model.joblib")
META = json.loads((BASE / "model" / "meta.json").read_text())
FEATURES = META["features"]
scaler = pipe.named_steps["scaler"]
clf = pipe.named_steps["clf"]

LABELS = {
    "vibration": "Vibration",
    "temperature": "Temperature",
    "pressure": "Pressure deviation",
    "current_draw": "Motor current",
    "hours_since_service": "Hours since service",
    "oil_quality": "Oil quality",
    "noise_level": "Noise level",
}

# Persona = how much each signal matters to that kind of user (1.0 = neutral)
PERSONAS = {
    "maintenance": {
        "name": "Maintenance Engineer",
        "focus": "Wear, lubrication and service intervals",
        "weights": {"vibration": 1.4, "temperature": 1.0, "pressure": 0.8, "current_draw": 0.8,
                    "hours_since_service": 1.5, "oil_quality": 1.5, "noise_level": 1.0},
    },
    "safety": {
        "name": "Safety Officer",
        "focus": "Overheating, pressure and hazardous conditions",
        "weights": {"vibration": 1.0, "temperature": 1.6, "pressure": 1.6, "current_draw": 1.2,
                    "hours_since_service": 0.7, "oil_quality": 0.7, "noise_level": 0.9},
    },
    "energy": {
        "name": "Energy Auditor",
        "focus": "Efficiency losses and abnormal power draw",
        "weights": {"vibration": 0.8, "temperature": 1.1, "pressure": 1.0, "current_draw": 1.8,
                    "hours_since_service": 0.8, "oil_quality": 0.9, "noise_level": 0.8},
    },
    "manager": {
        "name": "Plant Manager",
        "focus": "Downtime risk and planning",
        "weights": {k: 1.0 for k in FEATURES},
    },
}

ACTIONS = {
    "vibration": "Inspect bearings, alignment and mounting bolts.",
    "temperature": "Check cooling, ventilation and load; consider derating.",
    "pressure": "Inspect seals, valves and filters for blockage or leaks.",
    "current_draw": "Check motor windings, belt tension and mechanical load.",
    "hours_since_service": "Schedule preventive service soon.",
    "oil_quality": "Replace or filter the lubricant.",
    "noise_level": "Listen for gear/bearing faults; run an acoustic check.",
}


class Signals(BaseModel):
    vibration: float = Field(..., ge=0, le=30, description="mm/s")
    temperature: float = Field(..., ge=-20, le=250, description="°C")
    pressure: float = Field(..., ge=0, le=10, description="bar deviation")
    current_draw: float = Field(..., ge=0, le=100, description="A")
    hours_since_service: float = Field(..., ge=0, le=10000, description="h")
    oil_quality: float = Field(..., ge=0, le=100, description="%")
    noise_level: float = Field(..., ge=30, le=140, description="dB")


app = FastAPI(title="PulseGuard", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z))


def analyse(signals: Signals, persona_key: str | None = None):
    row = [[getattr(signals, f) for f in FEATURES]]
    scaled = scaler.transform(row)[0]
    coefs = clf.coef_[0]
    b0 = float(clf.intercept_[0])
    contribs = {f: float(c * x) for f, c, x in zip(FEATURES, coefs, scaled)}

    base_risk = float(pipe.predict_proba(row)[0][1])

    risk = base_risk
    persona = None
    if persona_key:
        persona = PERSONAS[persona_key]
        z = b0 + sum(persona["weights"][f] * contribs[f] for f in FEATURES)
        risk = sigmoid(z)

    ranked = sorted(contribs.items(), key=lambda kv: kv[1], reverse=True)
    drivers = [{"signal": k, "label": LABELS[k], "impact": round(v, 3)} for k, v in ranked if v > 0.05][:3]
    healthy = [{"signal": k, "label": LABELS[k], "impact": round(v, 3)} for k, v in ranked[::-1] if v < -0.05][:3]

    band = "LOW" if risk < 0.30 else "MEDIUM" if risk < 0.60 else "HIGH"
    health_score = round((1 - risk) * 100, 1)

    recs = []
    if band == "HIGH":
        recs.append("Plan maintenance immediately; avoid running at full load.")
    elif band == "MEDIUM":
        recs.append("Schedule an inspection within the next maintenance window.")
    else:
        recs.append("Machine looks healthy; continue routine monitoring.")
    recs += [ACTIONS[d["signal"]] for d in drivers[:2]]

    return {
        "failure_risk": round(risk, 4),
        "health_score": health_score,
        "risk_band": band,
        "baseline_risk": round(base_risk, 4),
        "persona": persona["name"] if persona else None,
        "persona_focus": persona["focus"] if persona else None,
        "top_risk_drivers": drivers,
        "healthy_signals": healthy,
        "contributions": {LABELS[k]: round(v, 3) for k, v in contribs.items()},
        "recommendations": recs,
    }


@app.get("/api")
def root():
    return {"status": "ok", "service": "PulseGuard", "docs": "/docs", "ui": "/ui"}


@app.get("/personas")
def personas():
    return {k: {"name": v["name"], "focus": v["focus"], "weights": v["weights"]} for k, v in PERSONAS.items()}


@app.post("/score")
def score(signals: Signals):
    return analyse(signals)


@app.post("/score/persona/{key}")
def score_persona(key: str, signals: Signals):
    if key not in PERSONAS:
        raise HTTPException(404, f"Unknown persona '{key}'. Options: {list(PERSONAS)}")
    return analyse(signals, key)


@app.get("/model-info")
def model_info():
    return META


app.mount("/ui", StaticFiles(directory=BASE / "frontend", html=True), name="ui")


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(BASE / "frontend" / "index.html")
