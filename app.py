from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from datetime import datetime
import joblib

from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

app = FastAPI(title="AI Message Triage System")

vectorizer = joblib.load("models/vectorizer.joblib")
model = joblib.load("models/model.joblib")

CONFIDENCE_THRESHOLD = 0.7

DATABASE_URL = "sqlite:///./tickets.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class Ticket(Base):
    __tablename__ = "tickets"

    id = Column(Integer, primary_key=True, index=True)
    sender = Column(String)
    text = Column(String)
    label = Column(String)
    confidence = Column(Float)
    status = Column(String, default="open")
    created_at = Column(String)
    resolved_at = Column(String, nullable=True)

Base.metadata.create_all(bind=engine)

class PredictRequest(BaseModel):
    text: str

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

class TicketListItem(BaseModel):
    id: int
    from_: str = Field(..., alias="from")
    label: str
    status: str

class ResolveRequest(BaseModel):
    status: str

class ResolveResponse(BaseModel):
    id: int
    status: str
    resolved_at: str

def predict_label(text: str):
    X = vectorizer.transform([text])
    proba = model.predict_proba(X)[0]
    confidence = float(max(proba))
    label = model.classes_[proba.argmax()]
    return label, confidence

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ml/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    label, confidence = predict_label(req.text)
    return {"label": label, "confidence": round(confidence, 3)}

@app.post("/messages/ingest", response_model=TicketResponse)
def ingest(req: IngestRequest):
    db = SessionLocal()
    label, confidence = predict_label(req.text)

    ticket = Ticket(
        sender=req.from_,
        text=req.text,
        label=label,
        confidence=round(confidence, 3),
        status="open",
        created_at=datetime.utcnow().isoformat() + "Z"
    )

    db.add(ticket)
    db.commit()
    db.refresh(ticket)
    db.close()

    return {
        "id": ticket.id,
        "from": ticket.sender,
        "text": ticket.text,
        "label": ticket.label,
        "confidence": ticket.confidence,
        "status": ticket.status,
        "created_at": ticket.created_at,
        "triage_required": confidence < CONFIDENCE_THRESHOLD
    }

@app.get("/tickets", response_model=List[TicketListItem])
def list_tickets(label: str | None = None, status: str | None = None):
    db = SessionLocal()
    query = db.query(Ticket)

    if label:
        query = query.filter(Ticket.label == label)
    if status:
        query = query.filter(Ticket.status == status)

    tickets = query.all()
    db.close()

    return [
        {
            "id": t.id,
            "from": t.sender,
            "label": t.label,
            "status": t.status
        }
        for t in tickets
    ]

@app.patch("/tickets/{ticket_id}", response_model=ResolveResponse)
def resolve(ticket_id: int, req: ResolveRequest):
    db = SessionLocal()
    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()

    if not ticket:
        db.close()
        raise HTTPException(status_code=404, detail="Ticket not found")

    ticket.status = req.status
    ticket.resolved_at = datetime.utcnow().isoformat() + "Z"

    db.commit()
    db.refresh(ticket)
    db.close()

    return {
        "id": ticket.id,
        "status": ticket.status,
        "resolved_at": ticket.resolved_at
    }
