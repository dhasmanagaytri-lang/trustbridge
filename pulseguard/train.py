"""Generate a synthetic machine-sensor dataset and train the PulseGuard model.

Run:  python train.py
Outputs: model/model.joblib, model/meta.json
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RNG = np.random.default_rng(42)
N = 4000

FEATURES = {
    # name: (label, min, max, unit, higher_is_riskier)
    "vibration": ("Vibration", 0.5, 12.0, "mm/s", True),
    "temperature": ("Temperature", 30.0, 120.0, "°C", True),
    "pressure": ("Pressure deviation", 0.0, 3.0, "bar", True),
    "current_draw": ("Motor current", 5.0, 40.0, "A", True),
    "hours_since_service": ("Hours since service", 0.0, 2000.0, "h", True),
    "oil_quality": ("Oil quality", 0.0, 100.0, "%", False),
    "noise_level": ("Noise level", 60.0, 110.0, "dB", True),
}


def generate() -> pd.DataFrame:
    data = {}
    for key, (_, lo, hi, _, _) in FEATURES.items():
        # skew toward healthy operation: beta distribution
        a, b = (2, 5) if FEATURES[key][4] else (5, 2)
        data[key] = lo + (hi - lo) * RNG.beta(a, b, N)
    df = pd.DataFrame(data)

    # hidden "true" relationship used only to create labels
    z = (
        -0.6
        + 0.55 * (df.vibration - 4) / 2.5
        + 0.45 * (df.temperature - 65) / 15
        + 0.30 * (df.pressure - 1) / 0.6
        + 0.25 * (df.current_draw - 18) / 7
        + 0.60 * (df.hours_since_service - 700) / 450
        - 0.50 * (df.oil_quality - 65) / 18
        + 0.20 * (df.noise_level - 80) / 7
        + RNG.normal(0, 0.6, N)
    )
    p = 1 / (1 + np.exp(-z))
    df["failure_7d"] = (RNG.random(N) < p).astype(int)
    return df


def main():
    df = generate()
    os.makedirs("data", exist_ok=True)
    os.makedirs("model", exist_ok=True)
    df.to_csv("data/machine_data.csv", index=False)

    X = df[list(FEATURES)]
    y = df["failure_7d"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    pipe.fit(Xtr, ytr)

    proba = pipe.predict_proba(Xte)[:, 1]
    meta = {
        "model": "LogisticRegression",
        "preprocessing": "StandardScaler (z = (x - mean) / std)",
        "target": "failure_7d (1 = failure within 7 days)",
        "n_samples": int(N),
        "failure_rate": round(float(y.mean()), 3),
        "accuracy": round(float(accuracy_score(yte, proba > 0.5)), 3),
        "roc_auc": round(float(roc_auc_score(yte, proba)), 3),
        "features": list(FEATURES),
        "coefficients": {
            k: round(float(c), 4)
            for k, c in zip(FEATURES, pipe.named_steps["clf"].coef_[0])
        },
        "intercept": round(float(pipe.named_steps["clf"].intercept_[0]), 4),
        "data_note": "Synthetic data generated for demonstration; replace with real sensor logs.",
    }
    joblib.dump(pipe, "model/model.joblib")
    with open("model/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
