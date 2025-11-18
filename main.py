import os
import re
import random
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from database import db, create_document, get_documents
from schemas import EmailLog, VendorRequest, Ticket

# ----- Constants -----
VM_QUERIES = "VM-QUERIES"
VM_IN_PROCESS = "VM-IN-PROCESS"
VM_RESPONDED = "VM-RESPONDED"
VM_ESCALATED = "VM-ESCALATED"

GMAIL_MODE = os.getenv("GMAIL_MODE", "mock")  # mock | live
GEMINI_MODE = os.getenv("GEMINI_MODE", "mock")  # mock | live

app = FastAPI(title="Vendor Master Email POC API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----- Utility serialization -----
def serialize_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    if not doc:
        return doc
    d = dict(doc)
    if "_id" in d:
        d["_id"] = str(d["_id"])  # type: ignore
    # Convert datetimes
    for k, v in list(d.items()):
        if isinstance(v, datetime):
            d[k] = v.isoformat()
    return d


# ----- Inputs -----
class MockEmailIn(BaseModel):
    from_email: str = Field(..., description="Sender email address")
    subject: str = Field(...)
    body: str = Field(...)


class SeedVendorRequestIn(BaseModel):
    request_id: str
    vendor_name: str
    email: Optional[str] = None
    country: Optional[str] = None
    stage: str = Field("Initiated")
    required_documents: List[str] = Field(default_factory=list)
    received_documents: List[str] = Field(default_factory=list)
    remarks: Optional[str] = None


# ----- Health -----
@app.get("/")
def read_root():
    return {"message": "Vendor Master Email POC API running"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
            response["database_name"] = db.name
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"
    return response


# ----- Agent Config & Status -----
@app.get("/agent/config")
def agent_config():
    return {
        "gmail_mode": GMAIL_MODE,
        "gemini_mode": GEMINI_MODE,
        "labels": [VM_QUERIES, VM_IN_PROCESS, VM_RESPONDED, VM_ESCALATED],
    }


@app.get("/agent/status")
def agent_status():
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")
    pending = db["emaillog"].count_documents({
        "labels": {"$in": [VM_QUERIES]},
        "labels": {"$nin": [VM_IN_PROCESS]},
    })
    in_process = db["emaillog"].count_documents({"labels": {"$in": [VM_IN_PROCESS]}})
    responded = db["emaillog"].count_documents({"labels": {"$in": [VM_RESPONDED]}})
    escalated = db["emaillog"].count_documents({"labels": {"$in": [VM_ESCALATED]}})
    return {
        "pending": pending,
        "in_process": in_process,
        "responded": responded,
        "escalated": escalated,
        "gmail_mode": GMAIL_MODE,
        "gemini_mode": GEMINI_MODE,
    }


# ----- Seed data helpers -----
@app.post("/seed/vendors")
def seed_vendors(items: Optional[List[SeedVendorRequestIn]] = None):
    """Seed mock VendorRequest data for deterministic demo."""
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")

    defaults: List[SeedVendorRequestIn] = items or [
        SeedVendorRequestIn(
            request_id="VR-2025-0012",
            vendor_name="Acme Supplies LLC",
            email="ops@acme.com",
            country="US",
            stage="Compliance",
            required_documents=["W-9", "Bank Statement (<=3 months)", "Cancelled Cheque"],
            received_documents=["W-9"],
            remarks=None,
        ),
        SeedVendorRequestIn(
            request_id="VR-2025-0042",
            vendor_name="Globex Manufacturing",
            email="contact@globex.io",
            country="IN",
            stage="Master Data",
            required_documents=["PAN", "GST", "Cancelled Cheque"],
            received_documents=["PAN", "GST", "Cancelled Cheque"],
            remarks=None,
        ),
        SeedVendorRequestIn(
            request_id="VR-2025-0099",
            vendor_name="Wayne Logistics",
            email="ap@wayne-log.com",
            country="UK",
            stage="Rejected",
            required_documents=["Bank Letter"],
            received_documents=["Bank Letter"],
            remarks="Compliance negative news hit",
        ),
    ]

    inserted = []
    for it in defaults:
        # Upsert by request_id
        existing = db["vendorrequest"].find_one({"request_id": it.request_id})
        doc = it.model_dump()
        if existing:
            db["vendorrequest"].update_one({"_id": existing["_id"]}, {"$set": {**doc, "updated_at": datetime.now(timezone.utc)}})
            updated = db["vendorrequest"].find_one({"request_id": it.request_id})
            inserted.append(serialize_doc(updated))
        else:
            create_document("vendorrequest", VendorRequest(**doc))
            new_doc = db["vendorrequest"].find_one({"request_id": it.request_id})
            inserted.append(serialize_doc(new_doc))

    return {"count": len(inserted), "items": inserted}


# ----- Email ingestion (mock) -----
@app.post("/ingest/mock-email")
def ingest_mock_email(payload: MockEmailIn):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")

    log = EmailLog(
        thread_id=f"thr_{random.randint(100000,999999)}",
        message_id=f"msg_{random.randint(100000,999999)}",
        from_email=payload.from_email,
        subject=payload.subject,
        body=payload.body,
        labels=[VM_QUERIES],
    )
    create_document("emaillog", log)
    # Fetch back for return
    doc = db["emaillog"].find_one({"thread_id": log.thread_id})
    return {"message": "Email ingested", "item": serialize_doc(doc)}


# ----- Processing logic (mock agent + tools) -----
REQUEST_ID_RE = re.compile(r"VR-\d{4}-\d+")


def detect_intent(text: str) -> str:
    t = text.lower()
    if any(w in t for w in ["status", "track", "progress"]):
        return "status"
    if any(w in t for w in ["document", "docs", "pending", "missing"]):
        return "docs"
    if any(w in t for w in ["bank change", "change bank", "update bank", "account change"]):
        return "bank_change"
    if any(w in t for w in ["compliance", "rejection", "rejected"]):
        return "compliance"
    if any(w in t for w in ["ticket", "escalate", "escalation"]):
        return "ticket_status"
    return "no_match"


def extract_entities(text: str) -> Dict[str, Any]:
    ents: Dict[str, Any] = {}
    rid = REQUEST_ID_RE.search(text)
    if rid:
        ents["request_id"] = rid.group(0)
    email_match = re.search(r"[\w\.-]+@[\w\.-]+", text)
    if email_match:
        ents.setdefault("email", email_match.group(0))
    return ents


def generate_new_request_id() -> str:
    return f"VR-2025-{random.randint(1000, 9999)}"


def get_vendor_request_by_entities(entities: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if db is None:
        return None
    coll = db["vendorrequest"]
    if "request_id" in entities:
        d = coll.find_one({"request_id": entities["request_id"]})
        if d:
            return d
    q = {}
    if entities.get("email"):
        q["email"] = entities["email"]
    if entities.get("vendor_name"):
        q["vendor_name"] = entities["vendor_name"]
    if q:
        d = coll.find_one(q)
        if d:
            return d
    return None


def create_ticket(reason: str, request_id: Optional[str] = None, email: Optional[str] = None, vendor_name: Optional[str] = None) -> str:
    tid = f"TCK-{random.randint(100000, 999999)}"
    create_document(
        "ticket",
        Ticket(ticket_id=tid, request_id=request_id, email=email, vendor_name=vendor_name, reason=reason),
    )
    return tid


@app.post("/process/next")
def process_next_email():
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")

    coll = db["emaillog"]
    doc = coll.find_one({"labels": {"$in": [VM_QUERIES]}, "labels": {"$nin": [VM_IN_PROCESS]}})
    if not doc:
        return {"message": "No emails pending", "processed": False}

    # Mark in-process
    labels = set(doc.get("labels", []))
    labels.add(VM_IN_PROCESS)
    coll.update_one({"_id": doc["_id"]}, {"$set": {"labels": list(labels), "updated_at": datetime.now(timezone.utc)}})

    body = doc.get("body", "") + "\n\n" + doc.get("subject", "")

    # If GEMINI_MODE is live, this is where you'd call the LLM to classify
    intent = detect_intent(body)
    entities = extract_entities(body)

    vr = get_vendor_request_by_entities(entities) if intent != "bank_change" else None

    reply_subject = f"Re: {doc.get('subject', '')}"
    reply_body = ""
    resolution_type = "auto_resolved"
    escalated = False
    ticket_id: Optional[str] = None

    if intent == "status":
        if vr:
            stage = vr.get("stage", "Initiated")
            reply_body = (
                f"Hello,\n\nYour request {vr['request_id']} is currently in {stage} review; "
                f"no action needed from your side." if stage not in ["Rejected"] else
                f"Hello,\n\nYour request {vr['request_id']} was rejected. Reason: {vr.get('remarks','Not specified')}"
            )
            if stage == "Rejected":
                resolution_type = "info_request"
        else:
            reply_body = (
                "Hello,\n\nWe couldn't find a matching request. Please share your Request ID (e.g., VR-2025-0012) or Tax ID so we can locate your case."
            )
            resolution_type = "info_request"
    elif intent == "docs":
        if vr:
            req = set(vr.get("required_documents", []))
            rec = set(vr.get("received_documents", []))
            missing = list(req - rec)
            if missing:
                reply_body = (
                    f"Hello,\n\nFor request {vr['request_id']}, we are missing the following documents: "
                    + ", ".join(missing) + ". Please share them to proceed."
                )
                resolution_type = "info_request"
            else:
                reply_body = (
                    f"Hello,\n\nFor request {vr['request_id']}, all required documents are received. Processing continues."
                )
        else:
            reply_body = (
                "Hello,\n\nPlease share your Request ID (e.g., VR-2025-0012) so we can check pending documents."
            )
            resolution_type = "info_request"
    elif intent == "bank_change":
        # Create new bank-change case
        new_id = generate_new_request_id()
        vr_payload = VendorRequest(
            request_id=new_id,
            vendor_name=entities.get("vendor_name") or "Unknown Vendor",
            email=doc.get("from_email"),
            country=None,
            stage="Initiated",
            required_documents=["Bank Statement (<=3 months)", "Cancelled Cheque", "Authorized Signatory Letter"],
            received_documents=[],
            remarks="Bank change case created via email",
        )
        create_document("vendorrequest", vr_payload)
        reply_body = (
            f"Hello,\n\nWe have created a bank change request {new_id}. "
            "Please provide: Bank Statement (<=3 months), Cancelled Cheque, Authorized Signatory Letter."
        )
    elif intent == "compliance":
        if vr and vr.get("stage") == "Rejected":
            reply_body = (
                f"Hello,\n\nYour request {vr['request_id']} was rejected after compliance checks. "
                f"Summary: {vr.get('remarks','Policy constraints')}."
            )
        elif vr:
            reply_body = (
                f"Hello,\n\nYour request {vr['request_id']} is undergoing compliance review. We'll update you shortly."
            )
        else:
            reply_body = (
                "Hello,\n\nPlease share your Request ID so we can check the compliance status."
            )
            resolution_type = "info_request"
    elif intent == "ticket_status":
        # Create ticket if not exists to showcase escalation
        reason = "User requested escalation via email"
        ticket_id = create_ticket(reason=reason, request_id=(vr or {}).get("request_id"), email=doc.get("from_email"))
        reply_body = (
            f"Hello,\n\nWe've created a support ticket {ticket_id}. Our team will follow up."
        )
        escalated = True
    else:  # no_match
        reply_body = (
            "Hello,\n\nWe couldn't determine the request details. Please share your Request ID (e.g., VR-2025-0012) and registered vendor name to proceed."
        )
        resolution_type = "info_request"

    # Update email log with decision and labels
    labels = set(coll.find_one({"_id": doc["_id"]}).get("labels", []))
    labels.discard(VM_QUERIES)
    labels.discard(VM_IN_PROCESS)
    labels.add(VM_RESPONDED)
    if escalated:
        labels.add(VM_ESCALATED)

    update_fields = {
        "intent": intent,
        "entities": entities,
        "resolution_type": resolution_type,
        "escalated": escalated,
        "ticket_id": ticket_id,
        "reply_subject": reply_subject,
        "reply_body": reply_body,
        "labels": list(labels),
        "updated_at": datetime.now(timezone.utc),
    }
    coll.update_one({"_id": doc["_id"]}, {"$set": update_fields})
    updated = coll.find_one({"_id": doc["_id"]})

    # In a real integration, here we'd call Gmail API to send the reply on the thread

    return {"processed": True, "item": serialize_doc(updated)}


# ----- Agent Orchestration -----
class RunLoopIn(BaseModel):
    max_steps: int = Field(10, ge=1, le=100)


@app.post("/agent/run-loop")
def agent_run_loop(payload: RunLoopIn):
    """Run the agent repeatedly until queue drains or max_steps reached."""
    steps = 0
    processed = 0
    last_item = None
    while steps < payload.max_steps:
        res = process_next_email()
        steps += 1
        if not res.get("processed"):
            break
        processed += 1
        last_item = res.get("item")
    return {"steps": steps, "processed": processed, "last_item": last_item}


@app.post("/agent/run-once")
def agent_run_once():
    return process_next_email()


# ----- Gmail scaffolding (mock/live switch) -----
@app.get("/gmail/poll")
def gmail_poll():
    """List pending messages per Gmail-style label query. In live mode, swap with Gmail API calls."""
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")
    # Mock: read from emaillog with labels
    cursor = db["emaillog"].find({"labels": {"$in": [VM_QUERIES]}, "labels": {"$nin": [VM_IN_PROCESS]}}).limit(50)
    items = [
        {
            "threadId": d.get("thread_id"),
            "messageId": d.get("message_id"),
            "snippet": (d.get("body", "")[:100] or d.get("subject", "")),
            "from": d.get("from_email"),
            "subject": d.get("subject"),
        }
        for d in cursor
    ]
    return {"mode": GMAIL_MODE, "query": "label:VM-QUERIES -label:VM-IN-PROCESS is:unread", "items": items}


class GmailReplyIn(BaseModel):
    thread_id: str
    reply_subject: str
    reply_body: str


@app.post("/gmail/reply")
def gmail_reply(payload: GmailReplyIn):
    """Send a reply on a thread. In mock mode, just record to log."""
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")
    # Mock: append a synthetic log entry to show a reply was made
    existing = db["emaillog"].find_one({"thread_id": payload.thread_id})
    if not existing:
        raise HTTPException(status_code=404, detail="Thread not found")
    # No-op for mock (actual send would go here)
    return {"mode": GMAIL_MODE, "sent": True}


# ----- Views for UI -----
@app.get("/logs")
def list_logs(limit: int = 50):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")
    cursor = db["emaillog"].find().sort("updated_at", -1).limit(limit)
    return [serialize_doc(d) for d in cursor]


@app.get("/analytics/summary")
def analytics_summary():
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")
    coll = db["emaillog"]
    total = coll.count_documents({})
    auto_resolved = coll.count_documents({"resolution_type": "auto_resolved"})
    info_request = coll.count_documents({"resolution_type": "info_request"})
    escalated = coll.count_documents({"escalated": True})

    # Count by intent
    intents = {}
    for it in ["status", "docs", "bank_change", "compliance", "ticket_status", "no_match"]:
        intents[it] = coll.count_documents({"intent": it})

    return {
        "total": total,
        "auto_resolved": auto_resolved,
        "info_request": info_request,
        "escalated": escalated,
        "by_intent": intents,
    }


@app.get("/vendors")
def list_vendors():
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")
    cursor = db["vendorrequest"].find().sort("updated_at", -1)
    return [serialize_doc(d) for d in cursor]


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
