"""Deterministic two-agent AutoGen policy review with no external model calls."""
import asyncio
import json
from pydantic import BaseModel

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.messages import StructuredMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from ola_support_agent.autogen_mock import create_mock_autogen_client


class VerdictModel(BaseModel):
    approved: bool
    final_answer: str
    reason: str


async def run_review_stage(draft_answer: str, retrieved_context: str, inject_error: bool = False) -> VerdictModel:
    if inject_error:
        draft_answer += " Ola provides immediate full cash refunds without verification."
    unsafe_claim = "full cash refunds" in draft_answer.casefold() or "without verification" in draft_answer.casefold()
    has_context = bool(retrieved_context.strip())
    approved = has_context and not unsafe_claim
    revised = draft_answer if approved else (
        "I don't have enough information to confirm this from the supplied policy context."
        if not has_context else
        "Refunds require verification and are processed according to the applicable policy."
    )
    verdict = VerdictModel(
        approved=approved,
        final_answer=revised,
        reason="Draft is supported by the supplied context." if approved else "Removed an unsupported refund promise.",
    )
    replay = create_mock_autogen_client([
        "Reviewer: compare the draft's claims with the supplied policy context.",
        verdict.model_dump_json(),
    ], structured_output_type=VerdictModel)
    reviewer = AssistantAgent(
        "Policy_Compliance_Reviewer", model_client=replay,
        system_message="Review the draft against the retrieved policy context. Flag unsupported claims.",
    )
    editor = AssistantAgent(
        "Final_Editor", model_client=replay,
        system_message="Return the required structured verdict, approving supported drafts unchanged.",
        output_content_type=VerdictModel,
    )
    team = RoundRobinGroupChat(
        [reviewer, editor],
        termination_condition=MaxMessageTermination(max_messages=3),
        max_turns=2,
        custom_message_types=[StructuredMessage[VerdictModel]],
    )
    result = await team.run(task=f"Context:\n{retrieved_context}\n\nDraft:\n{draft_answer}")
    await replay.close()
    content = result.messages[-1].content
    if isinstance(content, VerdictModel):
        return content
    if hasattr(content, "content") and isinstance(content.content, VerdictModel):
        return content.content
    return VerdictModel.model_validate_json(str(content))


def demo() -> None:
    context = "Overcharging refunds go to the original payment method after verification, within 3-5 business days."
    draft = "Overcharging refunds go to the original payment method after verification, within 3-5 business days."
    for injected in (False, True):
        result = asyncio.run(run_review_stage(draft, context, inject_error=injected))
        print(json.dumps({"injected": injected, **result.model_dump()}))


if __name__ == "__main__":
    demo()
