# 🏥 AI Message Triage System

## 📌 Project Overview

This project implements an **AI-powered message triage system** for a hospital environment. Incoming patient messages are automatically classified into categories such as **appointments, billing, reports, and complaints** using a machine learning model, and managed through a REST API built with **FastAPI**.

The system persists tickets using **SQLite**, exposes clean REST endpoints, and flags low-confidence predictions for **manual triage**, simulating a real-world hospital support workflow.

---

## 🛠️ Tech Stack

* **Backend:** Python 3.10+, FastAPI, Uvicorn
* **Machine Learning:** scikit-learn (TF-IDF + Logistic Regression)
* **Storage:** SQLite (via SQLAlchemy)
* **Serialization & Validation:** Pydantic
* **Model Persistence:** joblib

---

## 📂 Project Structure

```
tulu-health-assignment/
├── app.py                  # FastAPI application (SQLite + ML)
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
.venv\Scripts\activate      # Windows
```

### 2️⃣ Install dependencies

```cmd
pip install -r requirements.txt
```

### 3️⃣ Train the ML model

```cmd
python train.py
```

This step:

* Trains a TF-IDF + Logistic Regression classifier
* Evaluates the model using per-class F1 and Macro-F1
* Saves trained artifacts to the `models/` directory

### 4️⃣ Run the FastAPI server

```cmd
python -m uvicorn app:app --reload
```

Server starts at:
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
  "confidence": 0.81
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
  "confidence": 0.78,
  "status": "open",
  "created_at": "2026-01-12T10:00:00Z",
  "triage_required": false
}
```

📌 If `confidence < 0.7`, the system sets `"triage_required": true`.

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

* **Dataset:** 100 synthetic, balanced hospital messages (25 per class)
* **Vectorizer:** TF-IDF (unigrams + bigrams, `min_df=2`)
* **Classifier:** Logistic Regression
* **Train/Validation Split:** 80/20 (stratified)

### 📊 Model Evaluation

**Classification Report:**

```
              precision    recall  f1-score   support

 appointment       0.67      0.80      0.73         5
 billing           1.00      0.80      0.89         5
 complaint         0.75      0.60      0.67         5
 reports           0.67      0.80      0.73         5

 accuracy                               0.75        20
 macro avg          0.77      0.75      0.75        20
 weighted avg       0.77      0.75      0.75        20
```

**Macro-F1 Score:** **0.753**

📌 Macro-F1 is used as the primary metric to ensure **balanced performance across all message categories**, since all classes are equally important in hospital triage.

---

## 🧠 Design Notes

* ML model is trained offline and loaded once at API startup
* TF-IDF + Logistic Regression chosen for interpretability and efficiency
* Confidence scores derived from class probabilities
* Low-confidence predictions are flagged for manual triage
* SQLite ensures persistence across server restarts
* Pydantic models are used for **both request validation and response serialization**

---

## ✅ Conclusion

This project demonstrates a complete **end-to-end ML-powered backend system**, covering dataset creation, model training and evaluation, API design, persistence with SQLite, and robust serialization using Pydantic. The system is modular, reproducible, and aligned with real-world hospital message triage requirements.

---

