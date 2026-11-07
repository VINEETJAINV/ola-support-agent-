"""Pydantic response schemas shared by the CrewAI and API layers."""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Citation(BaseModel):
    doc_id: str
    chunk_id: str
    score: float
    text_snippet: str


class CrewResponse(BaseModel):
    answer: str = Field(min_length=1)
    citations: List[Citation] = Field(default_factory=list)
    ticket_info: Optional[Dict[str, Any]] = None
    confidence: float = Field(ge=0.0, le=1.0)
    fallback_triggered: bool = False
    session_id: str = "crew-demo"
    turn_number: int = 1
