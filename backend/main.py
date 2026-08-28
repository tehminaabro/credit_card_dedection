"""
SecureTrans
credit card fraud detection
--------------------------
Loads the trained imblearn Pipeline (StandardScaler -> SMOTE -> LogisticRegression)
and threshold.json, and serves:
  - GET  /health          quick check that the model loaded correctly
  - POST /predict         single-transaction prediction (REST)
  - WS   /ws               streams simulated live transactions through the
                            real model, in the exact format the dashboard expects

Run with:
    uvicorn main:app --reload --port 8000
"""

import json
import time
import sqlite3
import asyncio
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "fraud_detection_model.pkl"
THRESHOLD_PATH = BASE_DIR / "threshold.json"
DATA_PATH = BASE_DIR / "creditcard_sample.csv"  # small sample, keeps the repo GitHub-friendly
DB_PATH = BASE_DIR / "fraud_history.db"

# Exact order the model was trained on — do not change this order.
FEATURE_ORDER = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

app = FastAPI(title="SecureTrans Fraud Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # tighten this to your frontend's domain before real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- SQLite setup: stores every transaction that's ever been scored ----
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            txn_id TEXT,
            ts REAL,
            amount REAL,
            risk REAL,
            is_fraud INTEGER,
            actual INTEGER
        )
    """)
    conn.commit()
    conn.close()

def log_transaction(txn_id, ts, amount, risk, is_fraud, actual):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO transactions (txn_id, ts, amount, risk, is_fraud, actual) VALUES (?, ?, ?, ?, ?, ?)",
        (txn_id, ts, amount, risk, int(is_fraud), actual)
    )
    conn.commit()
    conn.close()

init_db()

# ---- Load model + threshold once, at startup ----
model = joblib.load(MODEL_PATH)
with open(THRESHOLD_PATH) as f:
    THRESHOLD = json.load(f)["threshold"]


class Transaction(BaseModel):
    Time: float
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    Amount: float


def predict_transaction(row: dict) -> dict:
    """Run one transaction dict through the pipeline. Skips SMOTE automatically
    (imblearn Pipelines only apply samplers during .fit())."""
    X = pd.DataFrame([row], columns=FEATURE_ORDER)
    proba = float(model.predict_proba(X)[0, 1])
    return {
        "risk": round(proba, 4),
        "isFraud": bool(proba >= THRESHOLD),
    }


@app.get("/health")
def health():
    return {"status": "ok", "threshold": THRESHOLD, "features_expected": FEATURE_ORDER}


@app.post("/predict")
def predict(transaction: Transaction):
    """Single transaction prediction, e.g. for a manual 'check this transaction' form."""
    result = predict_transaction(transaction.model_dump())
    log_transaction(
        txn_id="manual", ts=time.time(), amount=transaction.Amount,
        risk=result["risk"], is_fraud=result["isFraud"], actual=None
    )
    return result


@app.get("/sample_transaction")
def sample_transaction(fraud: bool = False):
    """Returns one real transaction's features from the sample dataset —
    used by the dashboard's manual 'Test a known case' buttons."""
    df = pd.read_csv(DATA_PATH)
    subset = df[df["Class"] == (1 if fraud else 0)]
    row = subset.sample(n=1).iloc[0]
    return row[FEATURE_ORDER].to_dict()


@app.get("/history")
def get_history(limit: int = 50):
    """Returns the most recent logged transactions from SQLite."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM transactions ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Streams rows from your dataset one by one (simulating a live payment feed),
    running each one through the REAL model, and pushes the result to the dashboard."""
    await websocket.accept()
    try:
        df = pd.read_csv(DATA_PATH)
        idx = 0
        n = len(df)
        while True:
            try:
                row = df.iloc[idx % n]
                payload = row[FEATURE_ORDER].to_dict()
                result = predict_transaction(payload)

                # "Class" is the ground-truth label in the Kaggle dataset (0 = normal, 1 = fraud).
                # It's never sent to the model — only used here so the dashboard can show
                # live accuracy (prediction vs. actual).
                actual_label = int(row["Class"]) if "Class" in df.columns else None

                log_transaction(
                    txn_id=f"TX-{idx:06d}", ts=time.time(), amount=float(row["Amount"]),
                    risk=result["risk"], is_fraud=result["isFraud"], actual=actual_label
                )

                await websocket.send_json({
                    "id": f"TX-{idx:06d}",
                    "time": time.time(),
                    "amount": float(row["Amount"]),
                    "risk": result["risk"],
                    "isFraud": result["isFraud"],
                    "actual": actual_label,
                })
            except WebSocketDisconnect:
                raise
            except Exception as e:
                # Log and skip this row instead of killing the whole stream.
                print(f"[ws] skipped row {idx}: {e}")

            idx += 1
            await asyncio.sleep(1.4)  # pacing so it feels like a live feed
    except WebSocketDisconnect:
        pass
