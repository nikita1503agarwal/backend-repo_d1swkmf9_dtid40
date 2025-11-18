"""
Database Schemas for the Vendor Master Email POC

Each Pydantic model corresponds to a MongoDB collection (lowercased class name).

Collections:
- EmailLog: Stores each handled email/thread interaction for analytics and replay
- VendorRequest: Mocked vendor onboarding/change requests
- Ticket: Escalations created by the agent for special handling
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class EmailLog(BaseModel):
    """
    Stores normalized email interactions processed by the system
    Collection name: "emaillog"
    """
    thread_id: str = Field(..., description="Gmail threadId")
    message_id: str = Field(..., description="Gmail messageId of the latest incoming vendor email")
    from_email: str = Field(..., description="Sender email address")
    subject: str = Field(..., description="Email subject")
    body: str = Field(..., description="Plaintext email body")
    labels: List[str] = Field(default_factory=list, description="Current logical labels applied to the thread")

    # Agent understanding
    intent: Optional[str] = Field(None, description="Detected intent: status | docs | bank_change | compliance | ticket_status | no_match")
    entities: Dict[str, Any] = Field(default_factory=dict, description="Extracted entities like request_id, vendor_name, tax_id, country")

    # Decision & outcome
    resolution_type: Optional[str] = Field(None, description="auto_resolved | escalated | info_request")
    escalated: bool = Field(False, description="Whether a ticket/escalation was created")
    ticket_id: Optional[str] = Field(None, description="If escalated, ticket identifier")

    # Reply details
    reply_subject: Optional[str] = Field(None)
    reply_body: Optional[str] = Field(None)

    # Timestamps (set by DB helper)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class VendorRequest(BaseModel):
    """
    Mocked vendor onboarding/change requests
    Collection name: "vendorrequest"
    """
    request_id: str = Field(..., description="Business-facing request identifier, e.g., VR-2025-0012")
    vendor_name: str = Field(...)
    email: Optional[str] = Field(None)
    country: Optional[str] = Field(None)
    stage: str = Field(..., description="Initiated | Compliance | Finance | Master Data | Completed | Rejected")
    required_documents: List[str] = Field(default_factory=list)
    received_documents: List[str] = Field(default_factory=list)
    remarks: Optional[str] = Field(None)


class Ticket(BaseModel):
    """
    Escalations created by the agent
    Collection name: "ticket"
    """
    ticket_id: str = Field(...)
    request_id: Optional[str] = Field(None)
    vendor_name: Optional[str] = Field(None)
    email: Optional[str] = Field(None)
    reason: str = Field(...)
    status: str = Field("Open", description="Open | In Progress | Resolved | Closed")
    priority: str = Field("Normal", description="Low | Normal | High | Urgent")
    created_by: str = Field("agent", description="agent | system | user")
    notes: Optional[str] = Field(None)

# Add additional schemas as needed
