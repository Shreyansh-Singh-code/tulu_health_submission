# 🏥 AI Message Triage System

## 📌 Project Overview

This project implements an **AI-powered message triage system** for a hospital setting. Incoming patient messages are automatically classified into categories such as **appointments, billing, reports, and complaints** using a machine learning model, and managed through a REST API built with FastAPI.

The system helps streamline hospital operations by routing messages efficiently and flagging low-confidence predictions for manual triage.

---

## 🛠️ Tech Stack

* **Backend:** Python 3.10+, FastAPI, Uvicorn
* **Machine Learning:** scikit-learn (TF-IDF + Logistic Regression)
* **Storage:** In-memory (can be extended to SQLite)
* **Serialization:** Pydantic
* **Model Persistence:** joblib

---

## 📂 Project Structure

```
ai-message-triage/
├── app.py                  # FastAPI application
├── train.py                # ML training script
├── dataset/
│   └── messages.csv        # Synthetic dataset (100 rows)
├── models/
│   ├── model.joblib        # Trained classifier
│   └── vectorizer.joblib   # TF-IDF vectorizer
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1️⃣ Create & activate virtual environment (recommended)

```cmd
python -m venv .venv
.venv\Scripts\activate  
```

### 2️⃣ Install dependencies

```cmd
pip install -r requirements.txt
```

### 3️⃣ Train the ML model

```cmd
python train.py
```

This will:

* Train a TF-IDF + Logistic Regression classifier
* Print per-class precision, recall, F1-score
* Print **Macro-F1 score**
* Save model artifacts to `/models`

### 4️⃣ Run the FastAPI server

```cmd
python -m uvicorn app:app --reload
```

Server will start at:
👉 [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## 🔍 API Testing

FastAPI provides an interactive Swagger UI:

👉 **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**

---

### ✅ 1. Health Check

**GET** `/health`

Response:

```json
{ "status": "ok" }
```

---

### ✅ 2. Predict Message Category

**POST** `/ml/predict`

Request:

```json
{
  "text": "I want to book an appointment tomorrow"
}
```

Response:

```json
{
  "label": "appointment",
  "confidence": 0.86
}
```

---

### ✅ 3. Ingest Message (Create Ticket)

**POST** `/messages/ingest`

Request:

```json
{
  "from": "+971500000001",
  "text": "I have not received my medical report"
}
```

Response:

```json
{
  "id": 1,
  "from": "+971500000001",
  "text": "I have not received my medical report",
  "label": "reports",
  "confidence": 0.82,
  "status": "open",
  "created_at": "2026-01-12T10:00:00Z",
  "triage_required": false
}
```

📌 If `confidence < 0.7`, `triage_required` is set to `true`.

---

### ✅ 4. List Tickets

**GET** `/tickets?label=reports&status=open`

Response:

```json
[
  {
    "id": 1,
    "from": "+971500000001",
    "label": "reports",
    "status": "open"
  }
]
```

---

### ✅ 5. Resolve Ticket

**PATCH** `/tickets/{id}`

Request:

```json
{
  "status": "resolved"
}
```

Response:

```json
{
  "id": 1,
  "status": "resolved",
  "resolved_at": "2026-01-12T10:30:00Z"
}
```

---

## 🤖 Machine Learning Details

* **Dataset:** 100 synthetic, balanced patient messages (25 per class)
* **Vectorizer:** TF-IDF (unigrams + bigrams, min_df=2)
* **Classifier:** Logistic Regression
* **Train/Test Split:** 80/20 (stratified)

### 📊 Evaluation Metrics

* Per-class Precision, Recall, F1-score
* **Macro-F1 score** (primary metric)

Example output:

```
Appointment F1: 0.90
Billing F1: 0.90
Reports F1: 0.84
Complaint F1: 0.92
Macro F1: 0.89
```

Macro-F1 is used to ensure **balanced performance across all classes**, since all message types are equally important.

---

## 🧠 Design Notes

* Model is trained offline and loaded once at API startup
* Confidence scores are derived from prediction probabilities
* Low-confidence predictions are flagged for manual triage
* In-memory storage used for simplicity (can be extended to SQLite)

---

## ✅ Conclusion

This project demonstrates an end-to-end **ML-powered backend system**, covering dataset creation, model training, evaluation, and real-time inference via REST APIs. It is fully runnable, modular, and easily extensible for production use.

---

