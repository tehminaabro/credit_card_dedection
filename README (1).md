# SecureTrans — Credit Card Fraud Detection System

An end-to-end machine learning project: a trained fraud-detection model served
through a FastAPI backend (REST + WebSocket) and a live monitoring dashboard.

## How it works
- **Model**: `StandardScaler → SMOTE → LogisticRegression` pipeline, trained on
  the Kaggle Credit Card Fraud Detection dataset (V1–V28 PCA features, Time, Amount).
- **Threshold**: chosen via precision-recall analysis on a held-out validation
  set (F1-optimal), not an arbitrary guess — see `backend/main.py` / `threshold.json`.
- **Backend**: FastAPI serves `/predict` (single transaction) and `/ws`
  (streams simulated live transactions through the real model), and logs every
  scored transaction to a local SQLite database (`fraud_history.db`).
- **Frontend**: a live dashboard (`frontend/fraud_dashboard.html`) that
  connects to the backend over WebSocket and shows real-time KPIs, a
  transaction feed, fraud alerts, and live accuracy against ground-truth labels.

## Project structure
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

## Running locally
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

## Deployment
Backend is deployed on [Render](https://render.com). See `backend/README.md`
for setup notes.
