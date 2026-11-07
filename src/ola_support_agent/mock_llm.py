"""Mock LLM for testing Ola Support Agent without requiring actual LLM API calls.

Extends CrewAI's BaseLLM for deterministic, offline ReAct loop behavior.
"""
import os
import json
import re
from typing import Any, Dict, List, Optional, Union, Tuple, List
from datetime import datetime

os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

from crewai.llms.base_llm import BaseLLM
from crewai.utilities.types import LLMMessage
from crewai.tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict, PrivateAttr


OBSERVATION_MARKER = "Observation:"
TASK_LABEL = "Current Task:"
TASK_END_LABELS = ("This is the expected criteria", "\n\nBegin!")
RECORD_ID_PATTERN = re.compile(r"\bTKT-\d{4}\b")


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return "" if content is None else str(content)


def _normalize(messages: Union[str, List[Dict[str, Any]]]) -> List[Dict[str, str]]:
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    return [{"role": m.get("role", ""), "content": _content_text(m.get("content"))} for m in messages]


def _extract_task_text(messages: List[Dict[str, str]]) -> str:
    """Return the task text from the first user message."""
    for m in messages:
        if m.get("role") != "user":
            continue
        text = m.get("content", "")
        start = text.find("Current Task:")
        if start == -1:
            return text.strip()
        text = text[start + len("Current Task:"):]
        for label in ("This is the expected criteria", "\n\nBegin!"):
            end = text.find(label)
            if end != -1:
                text = text[:end]
        return text.strip()
    return ""


def extract_observations(messages: List[Dict[str, str]]) -> List[str]:
    """Return real tool results, read only from the model's own (assistant) turns."""
    results = []
    for m in messages:
        if m.get("role") != "assistant":
            continue
        text = m.get("content", "")
        idx = text.rfind("Observation:")
        if idx != -1:
            results.append(text[idx + len("Observation:"):].strip())
    return results


def _compose_from_context(messages: List[Dict[str, str]]) -> Optional[str]:
    context_marker = "This is the context you're working with:"
    response_marker = "\n\nProvide your complete response:"
    for message in reversed(messages):
        content = message.get("content", "")
        if context_marker not in content:
            continue
        context = content.split(context_marker, 1)[1]
        if response_marker in context:
            context = context.split(response_marker, 1)[0]
        answer = context.split("\n\nCitations:", 1)[0].strip()
        if answer.startswith("Answer:"):
            answer = answer[len("Answer:"):].strip()
        if not answer:
            continue

        ticket = re.search(
            r"Ticket: (TKT-\d{4})\nStatus: ([^\n]+).*?"
            r"Escalation Score: ([0-9.]+)\nEscalation Recommended: (True|False)",
            context,
            re.DOTALL,
        )
        if ticket:
            record_id, status, score, recommended = ticket.groups()
            escalation = "is recommended" if recommended == "True" else "is not recommended"
            answer += (
                f"\n\nTicket {record_id} is {status}. Its escalation score is "
                f"{score}; escalation {escalation}."
            )
        return answer
    return None


class MockLLM(BaseLLM):
    """Deterministic, offline LLM for CrewAI. Same input always gives same output."""

    model: str = "mock-llm"
    responder: Any = None
    _call_count: int = PrivateAttr(default=0)

    def __init__(self, **data: Any) -> None:
        data.setdefault("model", "mock-llm")
        super().__init__(**data)
        self._call_count = 0
        self.responder = self._default_responder

    @property
    def call_count(self) -> int:
        return self._call_count

    @staticmethod
    def _default_responder(task_text: str, observations: List[str]) -> str:
        if observations:
            return observations[-1]
        return task_text

    def call(
        self,
        messages: Union[str, List[Dict[str, Any]]],
        tools: Optional[List[Dict[str, Any]]] = None,
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Any]] = None,
        from_task: Any = None,
        from_agent: Any = None,
        response_model: Any = None,
    ) -> str:
        self._call_count += 1
        msgs = messages if isinstance(messages, list) else _normalize(messages)
        task_text = _extract_task_text(msgs)
        observations = extract_observations(msgs)
        agent_tools = getattr(from_agent, "tools", None) or []

        if not observations:
            # No observation yet: decide which tool to call based on agent's tools
            for tool in agent_tools:
                # Derive arguments from each tool's declared schema. The
                # helpers are instance methods, so call them through self.
                args = self._build_tool_args(self._get_tool_schema(tool), task_text)
                if args is not None:
                    return (
                        f"Thought: I need to use a tool to answer this.\n"
                        f"Action: {tool.name}\n"
                        f"Action Input: {json.dumps(args)}"
                    )

        if getattr(from_agent, "role", None) == "Response Composer" and not agent_tools:
            answer = _compose_from_context(msgs)
            if answer:
                return f"Thought: I now know the final answer\nFinal Answer: {answer}"

        # Has observations: produce final answer
        answer = self.responder(task_text, observations)
        return f"Thought: I now know the final answer\nFinal Answer: {answer}"

    def supports_function_calling(self) -> bool:
        return False

    def supports_stop_words(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 8192

    def _get_tool_schema(self, tool: Any) -> dict:
        schema_cls = getattr(tool, "args_schema", None)
        if schema_cls is None:
            return {}
        return schema_cls.model_json_schema()

    def _build_tool_args(self, schema: dict, task_text: str) -> dict | None:
        properties = schema.get("properties", {})
        required = schema.get("required", list(schema.get("properties", {})))
        args = {}
        for field in properties:
            if field == "record_id":
                match = RECORD_ID_PATTERN.search(task_text)
                if match:
                    args[field] = match.group(0)
            elif field in {"query", "question"}:
                if task_text:
                    args[field] = task_text
        if any(field not in args for field in required):
            return None
        return args


def create_mock_llm(**kwargs) -> "MockLLM":
    return MockLLM(**kwargs)
