# SecureTrans — Credit Card Fraud Detection System

**Author:** Temina Abro

An end-to-end machine learning project that goes beyond a Jupyter notebook —
a trained fraud-detection model is served through a real backend API and
displayed on a live monitoring dashboard, the way a real fintech fraud
system would work.

🔗 **Live Demo:** https://securetrans-fraud-detection-system.onrender.com
🔗 **Backend API docs:** https://credit-card-dedection.onrender.com/docs

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
## Model Evaluation

The model was evaluated on a held-out test set using the tuned decision
threshold (see `threshold.json`), rather than the default 0.5 cutoff.

**Confusion Matrix**

|                  | Predicted Normal | Predicted Fraud |
|------------------|------------------|------------------|
| **Actual Normal** | 56,603 (TN)      | 48 (FP)          |
| **Actual Fraud**  | 20 (FN)          | 75 (TP)          |

**Classification Report**

| Class          | Precision | Recall | F1-score | Support |
|----------------|-----------|--------|----------|---------|
| 0 (Normal)     | 1.00      | 1.00   | 1.00     | 56,651  |
| 1 (Fraud)      | 0.61      | 0.79   | 0.69     | 95      |
| **Accuracy**   |           |        | 1.00     | 56,746  |
| **Macro avg**  | 0.80      | 0.89   | 0.84     | 56,746  |
| **Weighted avg**| 1.00     | 1.00   | 1.00     | 56,746  |

**Why accuracy is misleading here:** fraud makes up only ~0.17% of the
dataset. A model that predicted "normal" for every transaction would still
score ~99.8% accuracy while catching zero fraud. For this reason, model
performance is judged on **precision, recall, and F1 for the fraud class**,
not overall accuracy.

**Interpreting the results:**
- **Recall = 0.79** — the model correctly identifies 75 of the 95 actual
  fraud cases in the test set, missing 20.
- **Precision = 0.61** — of all transactions the model flags as fraud
  (123 total), 61% are genuinely fraudulent; the remaining 48 are false
  alarms on legitimate transactions.
- **F1 = 0.69** — the harmonic mean of the two, used here as the metric
  the decision threshold was tuned to maximize.

**The precision/recall trade-off:** in fraud detection, missing a fraud
case (false negative) is usually far more costly than a false alarm
(false positive) — a blocked legitimate transaction is an inconvenience,
but an undetected fraud is a direct financial loss. The threshold in this
project was tuned to lean toward catching more fraud, which is why recall
(0.79) is notably higher than precision (0.61). A different business
context (e.g. one where false alarms damage customer trust more) could
justify moving the threshold the other way.

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

