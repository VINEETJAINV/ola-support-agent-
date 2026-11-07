"""
Tools for Ola Support Agent.
"""
import os
import json
import re
from typing import Dict, Any, Optional

os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

from .dataset import SUPPORT_TICKETS, SupportTicket


def compute_escalation_score(ticket: SupportTicket) -> float:
    """
    Escalation score formula:
    - Base score from escalated flag: 0.6 if escalated=True, 0.0 if False
    - Recency signal: normalized days_since_created (0-30 days mapped to 0-0.4)
      - More recent tickets (lower days) get higher recency score
      - recency_score = (30 - days_since_created) / 30 * 0.4
    - Total score = base + recency_signal, clamped to [0, 1]
    """
    base_score = 0.6 if ticket.escalated else 0.0
    recency_score = ((30 - ticket.days_since_created) / 30) * 0.4
    total_score = min(1.0, base_score + recency_score)
    return round(total_score, 4)


ESCALATION_THRESHOLD = 0.65


def check_support_ticket_status(record_id: str) -> Dict[str, Any]:
    """
    Look up a support ticket by record_id and return status, resolution_time_hours,
    and a designed escalation_score in [0, 1].
    
    Escalation score combines:
    - escalated flag (0.6 weight)
    - normalized recency from days_since_created (0.4 weight, inverted so recent = higher)
    
    Threshold for recommending escalation: 0.65
    At 0.65 this selects escalated tickets created within the last 26 days,
    prioritizing recent escalated cases for follow-up.
    """
    for ticket in SUPPORT_TICKETS:
        if ticket.record_id == record_id:
            escalation_score = compute_escalation_score(ticket)
            return {
                "record_id": ticket.record_id,
                "status": ticket.status,
                "resolution_time_hours": ticket.resolution_time_hours,
                "escalation_score": escalation_score,
                "escalation_recommended": escalation_score >= ESCALATION_THRESHOLD,
                "category": ticket.category,
                "days_since_created": ticket.days_since_created,
                "escalated": ticket.escalated
            }
    
    return {
        "error": f"Ticket {record_id} not found",
        "record_id": record_id,
        "status": None,
        "resolution_time_hours": None,
        "escalation_score": 0.0,
        "escalation_recommended": False
    }


def get_escalation_stats() -> Dict[str, Any]:
    """Get dataset statistics for escalation score distribution."""
    scores = [compute_escalation_score(t) for t in SUPPORT_TICKETS]
    escalated_scores = [s for t, s in zip(SUPPORT_TICKETS, scores) if t.escalated]
    non_escalated_scores = [s for t, s in zip(SUPPORT_TICKETS, scores) if not t.escalated]
    
    return {
        "total_tickets": len(SUPPORT_TICKETS),
        "escalated_count": len(escalated_scores),
        "non_escalated_count": len(non_escalated_scores),
        "escalation_threshold": ESCALATION_THRESHOLD,
        "score_distribution": {
            "min": min(scores),
            "max": max(scores),
            "mean": round(sum(scores) / len(scores), 4),
            "escalated_mean": round(sum(escalated_scores) / len(escalated_scores), 4) if escalated_scores else 0,
            "non_escalated_mean": round(sum(non_escalated_scores) / len(non_escalated_scores), 4) if non_escalated_scores else 0,
        },
        "tickets_above_threshold": sum(1 for s in scores if s >= ESCALATION_THRESHOLD),
        "formula": "score = (0.6 if escalated else 0.0) + ((30 - days_since_created) / 30) * 0.4",
        "threshold_justification": "0.65 requires escalated=True and days_since_created <= 26, prioritizing recent escalated cases"
    }


if __name__ == "__main__":
    import json
    stats = get_escalation_stats()
    print(json.dumps(stats, indent=2))
    
    print("\n--- Sample lookups ---")
    for ticket in SUPPORT_TICKETS[:3]:
        result = check_support_ticket_status(ticket.record_id)
        print(json.dumps(result, indent=2))
