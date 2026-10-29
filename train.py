"""Train the TrustBridge model and save it to model.joblib.

The dataset here is SYNTHETIC so the project runs out of the box.
To use real data, replace make_data() with a CSV load (columns = FEATURES + "trusted").
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURES = ["reliability", "transparency", "responsiveness", "community", "verification"]


def make_data(n=3000, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(10, 100, size=(n, len(FEATURES)))
    # hidden "true" importance of each signal, used only to generate labels
    true_w = np.array([0.30, 0.20, 0.15, 0.15, 0.20])
    score = X @ true_w + rng.normal(0, 6, n)
    y = (score > np.median(score)).astype(int)
    return pd.DataFrame(X, columns=FEATURES).assign(trusted=y)


def main():
    df = make_data()
    X_tr, X_te, y_tr, y_te = train_test_split(
        df[FEATURES], df["trusted"], test_size=0.2, random_state=42, stratify=df["trusted"]
    )
    scaler = StandardScaler().fit(X_tr)          # z = (x - mean) / std
    model = LogisticRegression(max_iter=1000).fit(scaler.transform(X_tr), y_tr)

    proba = model.predict_proba(scaler.transform(X_te))[:, 1]
    metrics = {
        "accuracy": round(accuracy_score(y_te, proba > 0.5), 4),
        "roc_auc": round(roc_auc_score(y_te, proba), 4),
        "train_rows": len(X_tr),
    }
    joblib.dump({"model": model, "scaler": scaler, "features": FEATURES, "metrics": metrics}, "model.joblib")
    print("Saved model.joblib", metrics)
    for f, c in zip(FEATURES, model.coef_[0]):
        print(f"  {f:15s} coef = {c:+.3f}")


if __name__ == "__main__":
    main()