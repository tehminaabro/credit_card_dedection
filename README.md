# SecureTrans — Credit Card Fraud Detection System

**Author:** Temina Abro

An end-to-end machine learning project that goes beyond a Jupyter notebook —
a trained fraud-detection model is served through a real backend API and
displayed on a live monitoring dashboard, the way a real fintech fraud
system would work.

🔗 **Live Demo:** _[Add your Render deployment link here once deployed]_

---

## 1. What this project does

Credit card companies need to flag suspicious transactions **the moment
they happen**, not after the fact. This project takes a trained machine
learning model and wraps it in a real, working system:

1. A transaction comes in (simulated here from real historical data)
2. The trained model scores it for fraud probability, in real time
3. The result is streamed instantly to a live dashboard over WebSocket
4. Every scored transaction is logged to a database for later review

---

## 2. Dataset

[Credit Card Fraud Detection dataset (Kaggle / ULB - Machine Learning Group)](https://www.kaggle.com/mlg-ulb/creditcardfraud)

- 284,807 transactions made by European cardholders, of which only 492
  (~0.17%) are fraudulent — a highly **imbalanced** classification problem.
- Features `V1`–`V28` are the result of a **PCA transformation** applied by
  the dataset creators for confidentiality — the original transaction
  details (merchant, location, etc.) are not disclosed. `Time` and `Amount`
  are the only untransformed features.
- `Class` is the target label (0 = normal, 1 = fraud).

## 3. Methodology / Logic used

**Pipeline:** `StandardScaler → SMOTE → LogisticRegression`
(built as a single `imblearn.pipeline.Pipeline`)

- **StandardScaler** — normalizes all 30 input features to a common scale,
  since `Amount` and `Time` are on very different scales from the PCA
  components.
- **SMOTE (Synthetic Minority Over-sampling Technique)** — addresses the
  severe class imbalance (fraud is <0.2% of the data) by generating
  synthetic fraud examples during training. Because it's inside an
  `imblearn` pipeline, SMOTE is applied **only during `.fit()`** — it never
  touches validation/test data or live predictions, which avoids data
  leakage.
- **Logistic Regression** — the final classifier, chosen for being fast,
  interpretable, and a strong baseline for this kind of tabular fraud data.

**Decision threshold:** rather than using the default 0.5 cutoff, the
threshold was **tuned using a precision-recall analysis** on a held-out
validation set, selecting the value that maximizes the F1-score (balancing
missed fraud against false alarms). This is stored in `threshold.json` and
loaded dynamically by the backend — see `backend/main.py`.

## 4. Tech stack & architecture

| Layer | Technology | Purpose |
|---|---|---|
| Model | scikit-learn + imbalanced-learn | Trained fraud classifier |
| Backend | **FastAPI** | REST (`/predict`) + **WebSocket** (`/ws`) API serving live predictions |
| Storage | **SQLite** | Logs every scored transaction (`fraud_history.db`) for history/audit |
| Frontend | HTML / CSS / JavaScript + Chart.js | Live dashboard: KPIs, transaction feed, fraud alerts, charts |
| Deployment | **Render** | Hosts the FastAPI backend (WebSocket-compatible) |

**Why WebSocket instead of just REST?** A REST endpoint only answers one
request at a time. Real fraud monitoring needs a continuous stream of
results pushed to the screen as transactions happen — WebSocket keeps a
persistent connection open so the backend can push predictions to the
dashboard instantly, without the frontend having to keep asking for updates.

## 5. Project structure
```
fraud-detection-app/
├── backend/
│   ├── main.py                  FastAPI app (REST + WebSocket + SQLite logging)
│   ├── requirements.txt
│   ├── fraud_detection_model.pkl
│   ├── threshold.json
│   └── creditcard_sample.csv    small sample used to simulate a live feed
├── frontend/
│   └── fraud_dashboard.html
└── README.md
```

## 6. Running locally
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
Then open `frontend/fraud_dashboard.html` in a browser (ideally via a local
server, e.g. VSCode's Live Server, rather than double-clicking the file).

- API docs: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health
- Transaction history: http://127.0.0.1:8000/history

## 7. Deployment

Backend is deployed on [Render](https://render.com), which supports Python
web services with WebSocket connections. See `backend/README.md` for setup
notes.

## 8. Author

Made by **Temina Abro** as an end-to-end machine learning submission.

