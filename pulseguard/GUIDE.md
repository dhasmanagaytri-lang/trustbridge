# PulseGuard
## Complete Class Presentation & Viva Preparation Guide

---

## 1. Introduction

**Presentation script**

Good morning everyone. Today I am presenting my project, PulseGuard. PulseGuard is an intelligent machine health and maintenance recommendation system for Industrial IoT. It analyses multiple sensor signals from a machine and predicts the probability of failure in the next seven days.

**Motivation**

Factories rely on motors, pumps and compressors. Unplanned breakdowns cause downtime and safety risks, while fixed-schedule maintenance wastes money. Most monitoring dashboards use fixed alarm thresholds on one sensor at a time. Machine health is really a combination of many signals, so PulseGuard combines them using machine learning.

---

## 2. Problem Statement

**Problem**

- Single-sensor threshold alarms miss combined, gradual degradation.
- Reactive maintenance causes unplanned downtime.
- Everyone sees the same alert, though a safety officer, maintenance engineer and energy auditor care about different things.
- Alarms rarely explain *why* a machine is at risk.

**Example**

A motor with slightly high temperature, ageing oil and many hours since service may trigger no single alarm, yet be close to failure. Also, a hot, high-pressure machine matters more to a Safety Officer than to an Energy Auditor.

**Core idea**

Instead of *Health = one threshold per sensor*, PulseGuard treats health as a combination of signals and uses ML to turn them into a failure probability.

---

## 3. Meaning of PulseGuard

*Pulse* = the continuous heartbeat of sensor data from a machine. *Guard* = protecting the machine and people by catching problems early.

---

## 4. Main Objectives

1. **Health scoring** - failure probability and 0-100 health score from 7 signals.
2. **Personalization** - different roles see risk through their own priorities.
3. **Explainability** - show which signals pushed risk up or down.
4. **Recommendation** - actionable maintenance advice based on risk band and top drivers.

---

## 5. How PulseGuard Works

**Workflow**

User -> Sensor signals -> Validation -> Feature scaling -> Logistic Regression -> Failure probability -> Feature contributions -> Persona weighting -> Risk band -> Recommendation

The user enters sensor readings and picks a persona. FastAPI validates the input, scales it with the same scaler used in training, and the model outputs a failure probability. Each feature's contribution is computed, re-weighted by persona, and converted to a risk band and recommendations.

---

## 6. Sensor Signals

A signal is one measurable piece of evidence about machine condition.

| Signal | Unit | Direction |
|---|---|---|
| Vibration | mm/s | higher = riskier |
| Temperature | °C | higher = riskier |
| Pressure deviation | bar | higher = riskier |
| Motor current | A | higher = riskier |
| Hours since service | h | higher = riskier |
| Oil quality | % | **lower** = riskier |
| Noise level | dB | higher = riskier |

**Key point:** multiple signals give a richer picture than any single threshold.

---

## 7. Persona System

A persona is a type of user with specific priorities.

| Persona | Focus | Emphasised signals |
|---|---|---|
| Maintenance Engineer | Wear, lubrication, service intervals | Hours since service, oil quality, vibration |
| Safety Officer | Overheating, pressure, hazards | Temperature, pressure, current |
| Energy Auditor | Efficiency, abnormal power draw | Motor current |
| Plant Manager | Downtime and planning | Balanced (all weights 1.0) |

**Implementation:** each persona has a weight per signal. The model's per-signal contribution to the logit is multiplied by that weight before the sigmoid is applied.

**Demo result (same machine: temp 98 °C, pressure deviation 2.4 bar, everything else moderate):**
Maintenance 46% (Medium), Plant Manager 57% (Medium), Energy 63% (High), Safety 86% (High).

---

## 8. Machine Learning Model - Logistic Regression

Despite the name, it is a classification model. It outputs the probability of failure.

**Formula**

P(failure) = 1 / (1 + e^(-z)), where z = b0 + b1x1 + b2x2 + ... + bnxn

x = scaled sensor values, b = learned coefficients, b0 = intercept.

**Why Logistic Regression?** Fast, simple, probabilistic, and its coefficients are directly interpretable, which is what makes the explainability feature possible.

**Trained coefficients (on standardised features):**

| Signal | Coefficient |
|---|---|
| Temperature | +0.49 |
| Hours since service | +0.39 |
| Vibration | +0.35 |
| Current | +0.28 |
| Pressure | +0.26 |
| Noise | +0.20 |
| Oil quality | -0.40 |

Intercept = -1.55. Test accuracy about 79%, ROC-AUC about 0.72 (data has deliberate noise, so it is not unrealistically perfect). Failure rate in the data is about 21%.

---

## 9. Feature Scaling

Sensors have very different ranges (hours up to 2000, pressure up to 3). Scaling puts them on a comparable scale so large-valued features do not dominate and coefficients can be compared.

z = (x - mean) / std

PulseGuard uses `StandardScaler` inside a scikit-learn `Pipeline`, so the exact training scaling is applied at prediction time.

---

## 10. Explainability

For each signal: contribution = coefficient x scaled value.
Positive contribution pushes failure risk up; negative pushes it down.

**Example output**

Health score 62. Risk drivers: Temperature, Hours since service. Healthy signals: Oil quality, Noise.

**Presentation line:** PulseGuard does not treat the prediction as a black box. It shows which signals raised or lowered the risk, shown as red and green bars in the UI.

---

## 11. Backend - FastAPI

FastAPI exposes the model as a REST API. The frontend sends JSON; the backend validates it (Pydantic enforces realistic sensor ranges), scales, predicts, explains and returns JSON.

Frontend -> HTTP request -> FastAPI -> scaler -> model -> explanation + persona logic -> JSON -> Frontend

---

## 12. API Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /api` | Check that the API is running |
| `GET /personas` | Persona names, focus and weights |
| `POST /score` | Baseline failure risk |
| `POST /score/persona/{key}` | Persona-weighted risk and recommendations |
| `GET /model-info` | Model type, metrics, coefficients |
| `/docs` | Auto-generated Swagger UI |

---

## 13. Frontend User Flow

Open site -> set sensor sliders -> choose persona -> Analyse machine -> health score, risk band, explanation bars, recommendations.

**Demo tip:** run the same readings under two personas (Safety vs Maintenance) to show personalization.

---

## 14. Technology Stack

- Frontend: HTML, CSS, JavaScript (single page)
- Backend: Python, FastAPI, Uvicorn
- ML: scikit-learn Logistic Regression
- Preprocessing: StandardScaler (Pipeline)
- Data: synthetic sensor dataset (4000 samples) generated with NumPy/Pandas
- Storage: joblib model file + JSON metadata

---

## 15. Example Decision

Inputs: vibration 9 mm/s, temperature 105 °C, pressure deviation 2.5 bar, current 34 A, hours since service 1800, oil quality 15%, noise 100 dB.

Result: health score about 0, HIGH risk. Main drivers: temperature, hours since service, oil quality. Recommendation: plan maintenance immediately, check cooling, schedule service, change lubricant.

A healthy machine (vibration 2, temp 50, oil 90%, 150 h since service) scores about 97 (LOW risk).

---

## 16. What Happens Behind the Scenes

1. UI collects slider values and persona.
2. Browser POSTs JSON to `/score/persona/{key}`.
3. Pydantic validates ranges (422 on invalid input, 404 on unknown persona).
4. Values are scaled using the training mean/std.
5. Logistic Regression computes the logit and probability.
6. Per-feature contributions are calculated.
7. Persona weights re-scale the contributions and the probability is recomputed.
8. Risk band: Low below 30%, Medium 30-60%, High above 60%.
9. Top drivers are mapped to maintenance actions.
10. JSON is returned and the UI renders it.

---

## 17. Why Not Just Use Fixed Thresholds or an Average?

Thresholds look at one sensor at a time and cannot see combined degradation. A plain average treats all signals as equally important (and oil quality points the opposite direction). ML learns each signal's weight and direction from data.

---

## 18. What Makes PulseGuard Different

1. **Multi-signal** - health from seven combined indicators.
2. **Personalized** - role-based interpretation.
3. **Explainable** - per-signal contributions.
4. **ML-powered and actionable** - trained model plus concrete recommendations.

---

## 19. Limitations

- Trained on **synthetic** data; real deployment needs real labelled failure logs.
- Logistic Regression is linear and cannot capture complex interactions (e.g. temperature x load).
- Static readings only, no time-series trends.
- Persona weights are hand-set, not learned; persona scores are an interpretation layer and are not separately calibrated.
- Failure labels in real plants are rare (class imbalance).

---

## 20. Future Scope

- Real sensor data via MQTT / OPC UA from PLCs or edge gateways.
- Time-series features (rolling means, trends) and models such as Random Forest, XGBoost, LSTM.
- SHAP or LIME explanations.
- Remaining Useful Life (RUL) regression.
- Anomaly detection for sensor faults or tampering.
- Alerts via SMS/email and a maintenance ticketing integration.
- Learn persona weights from user feedback.

---

## 21. Recommended PPT Structure

1. Title - PulseGuard: Persona-Aware Machine Health & Maintenance Recommendation (name, roll number, branch, college)
2. Problem statement
3. Proposed solution
4. Objectives
5. System architecture (flow diagram)
6. Technology stack
7. Machine learning (Logistic Regression + scaling)
8. Persona system
9. Explainability
10. Live demo
11. Limitations and future scope
12. Conclusion and questions

---

## 22. Viva Questions & Answers

**Q1. What is PulseGuard?**
An ML-based machine health system that combines seven sensor signals, predicts the probability of failure within seven days using Logistic Regression, explains the result per signal, and gives role-specific recommendations.

**Q2. Why machine learning instead of thresholds?**
Failure depends on combinations of signals. ML learns the weight and direction of each signal from data instead of relying on manual thresholds.

**Q3. Why Logistic Regression?**
It is simple, fast, outputs probabilities and has interpretable coefficients, which supports explainability.

**Q4. Why feature scaling?**
Sensor ranges differ widely. Scaling prevents large-valued features dominating and makes coefficients comparable.

**Q5. What is a persona and how is it implemented?**
A user role with priorities. Each persona has per-signal weights applied to the model's feature contributions before the final probability.

**Q6. How is explainability achieved?**
Contribution = coefficient x scaled feature value. Positive values raise risk, negative values lower it.

**Q7. Why is oil quality's coefficient negative?**
Higher oil quality means a healthier machine, so it reduces failure risk.

**Q8. Is your data real?**
No, it is synthetic, generated from a hidden formula plus noise, to demonstrate the pipeline. The same code works with real logs.

**Q9. Why is accuracy only about 79%?**
Noise was added on purpose to mimic real-world uncertainty. Accuracy alone is also not enough for imbalanced data, so ROC-AUC is reported too.

**Q10. How does the frontend talk to the model?**
Through REST. The browser sends JSON to FastAPI, which scales the input, runs the model and returns JSON.

**Q11. How would you use it in a real plant?**
Stream readings from PLCs/sensors through an edge gateway (MQTT/OPC UA), call the API periodically, and raise alerts on risk-band changes.

**Q12. What would you improve?**
Real data, time-series features, Random Forest/XGBoost, SHAP, RUL prediction and learned persona weights.

---

## 23. One-Minute Project Explanation

Good morning everyone. PulseGuard is a persona-aware machine health and maintenance recommendation system for Industrial IoT. The problem is that traditional monitoring relies on fixed alarm thresholds on single sensors, which miss combined degradation and give everyone the same alert.

PulseGuard combines seven signals: vibration, temperature, pressure deviation, motor current, hours since service, oil quality and noise. These are scaled and passed to a Logistic Regression model, which predicts the probability of failure within seven days. The system also calculates each signal's contribution, so users see why a machine is at risk.

Because different roles care about different things, we use personas: a maintenance engineer, safety officer, energy auditor and plant manager each interpret the same readings through their own priorities. The backend is Python with FastAPI, exposing endpoints for scoring, personas and model information, and the web frontend shows the health score, explanation and recommendations.

The strengths are multi-signal analysis, personalization, explainability and actionable output. In future we can add real sensor data, time-series models, SHAP and remaining-useful-life prediction.

---

## 24. Final Memory Trick

**Problem -> Signals -> Scaling -> Logistic Regression -> Failure Risk -> Explanation -> Persona -> Recommendation**

Closing: PulseGuard shows how machine learning can combine multiple sensor signals to give personalized, explainable maintenance decisions. The focus is not only predicting failure, but helping people understand and act on the reasons.

Thank you. I am happy to answer any questions.
