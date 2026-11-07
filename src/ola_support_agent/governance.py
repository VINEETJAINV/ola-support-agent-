"""Application and runtime governance controls for the capstone."""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List


class RiskLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


@dataclass(frozen=True)
class ToolPermission:
    allowed: tuple[str, ...]


class LeastAutonomyEnforcer:
    PERMISSIONS = {
        "Retrieval Agent": ToolPermission(("rag_lookup",)),
        "Lookup Agent": ToolPermission(("check_support_ticket_status",)),
        "Response Composer": ToolPermission(()),
    }

    @classmethod
    def check_permission(cls, agent_name: str, tool_name: str) -> bool:
        permission = cls.PERMISSIONS.get(agent_name)
        return permission is not None and tool_name in permission.allowed

    @classmethod
    def demonstrate(cls) -> None:
        for agent, tool in (("Lookup Agent", "check_support_ticket_status"), ("Retrieval Agent", "check_support_ticket_status"), ("Response Composer", "rag_lookup")):
            print(f"{agent} -> {tool}: {'allowed' if cls.check_permission(agent, tool) else 'blocked'}")


class CostBudgetEnforcer:
    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens

    @staticmethod
    def estimate_tokens(text: str) -> int:
        return (len(text) + 3) // 4

    def check_budget(self, estimated_tokens: int) -> Dict[str, object]:
        allowed = estimated_tokens <= self.max_tokens
        return {"allowed": allowed, "estimated_tokens": estimated_tokens, "max_tokens": self.max_tokens,
                "reason": "within budget" if allowed else f"request exceeds {self.max_tokens}-token cap"}

    def demonstrate(self) -> None:
        print(self.check_budget(500))
        print(self.check_budget(5000))


class RiskClassifier:
    @staticmethod
    def classify() -> Dict[str, str]:
        return {"risk_level": RiskLevel.MEDIUM.value,
                "justification": "Customer support is Medium risk under the supplied scheme; incorrect policy guidance or ticket handling can affect customers, so the system uses retrieval grounding, PII masking, review, and audit logs."}


if __name__ == "__main__":
    LeastAutonomyEnforcer.demonstrate()
    print(RiskClassifier.classify())
    CostBudgetEnforcer().demonstrate()
