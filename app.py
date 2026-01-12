from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime
import joblib

app = FastAPI(title="AI Message Triage System")

vectorizer = joblib.load("models/vectorizer.joblib")
model = joblib.load("models/model.joblib")

tickets = []
current_id = 1

CONFIDENCE_THRESHOLD = 0.7


class PredictRequest(BaseModel):
    text: str = Field(..., example="I want to book an appointment")

class PredictResponse(BaseModel):
    label: str
    confidence: float

class IngestRequest(BaseModel):
    from_: str = Field(..., alias="from")
    text: str

class TicketResponse(BaseModel):
    id: int
    from_: str = Field(..., alias="from")
    text: str
    label: str
    confidence: float
    status: str
    created_at: str
    triage_required: bool

class ResolveRequest(BaseModel):
    status: str


def predict_label(text: str):
    X = vectorizer.transform([text])
    probabilities = model.predict_proba(X)[0]
    confidence = float(max(probabilities))
    label = model.classes_[probabilities.argmax()]
    return label, confidence

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/ml/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    label, confidence = predict_label(req.text)
    return {"label": label, "confidence": round(confidence, 3)}

@app.post("/messages/ingest", response_model=TicketResponse)
def ingest_message(req: IngestRequest):
    global current_id

    label, confidence = predict_label(req.text)

    ticket = {
        "id": current_id,
        "from": req.from_,
        "text": req.text,
        "label": label,
        "confidence": round(confidence, 3),
        "status": "open",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "triage_required": confidence < CONFIDENCE_THRESHOLD
    }

    tickets.append(ticket)
    current_id += 1
    return ticket

@app.get("/tickets")
def list_tickets(label: str | None = None, status: str | None = None):
    result = tickets
    if label:
        result = [t for t in result if t["label"] == label]
    if status:
        result = [t for t in result if t["status"] == status]

    return [
        {
            "id": t["id"],
            "from": t["from"],
            "label": t["label"],
            "status": t["status"]
        }
        for t in result
    ]

@app.patch("/tickets/{ticket_id}")
def resolve_ticket(ticket_id: int, req: ResolveRequest):
    for t in tickets:
        if t["id"] == ticket_id:
            t["status"] = req.status
            t["resolved_at"] = datetime.utcnow().isoformat() + "Z"
            return {
                "id": t["id"],
                "status": t["status"],
                "resolved_at": t["resolved_at"]
            }
    raise HTTPException(status_code=404, detail="Ticket not found")
