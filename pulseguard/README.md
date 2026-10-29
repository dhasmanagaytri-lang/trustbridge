# PulseGuard

Persona-aware machine health scoring and maintenance recommendation system.
Built on the same pattern as TrustBridge (multi-signal input -> scaling -> Logistic Regression -> explainability -> personas -> recommendation), applied to Industrial IoT condition monitoring.

## Run it

```bash
pip install -r requirements.txt
python train.py                      # generates data/ and model/
python -m uvicorn app.main:app --reload
```

Open http://localhost:8000 for the UI and http://localhost:8000/docs for the interactive API docs.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api` | Health check |
| GET | `/personas` | List personas, their focus and signal weights |
| POST | `/score` | Baseline failure risk from the 7 sensor signals |
| POST | `/score/persona/{key}` | Persona-weighted risk + recommendations (`maintenance`, `safety`, `energy`, `manager`) |
| GET | `/model-info` | Model type, metrics, coefficients, intercept |

Example body:

```json
{"vibration":4,"temperature":98,"pressure":2.4,"current_draw":16,
 "hours_since_service":300,"oil_quality":80,"noise_level":75}
```

## Structure

```
train.py            synthetic data + training (StandardScaler + LogisticRegression)
app/main.py         FastAPI backend
frontend/index.html single-page UI (served by FastAPI at / and /ui)
model/              model.joblib, meta.json (created by train.py)
data/               machine_data.csv (created by train.py)
GUIDE.md            presentation script, viva Q&A, one-minute explanation
```

## Notes

- Data is synthetic. Swap `data/machine_data.csv` for real sensor logs (same columns plus `failure_7d`) and re-run `train.py`.
- Persona scores re-weight each signal's contribution to the logit; they are an interpretation layer, not separately trained models.
- Deployment: host the API on Render/Railway/Fly (Vercel's serverless limits make the scikit-learn bundle awkward), or keep it local for the demo.
