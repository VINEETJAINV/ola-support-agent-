"""Process-local conversation memory backed by LangChain message history."""
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory


class SessionMemory:
    def __init__(self):
        self.sessions: Dict[str, InMemoryChatMessageHistory] = {}
        runnable = RunnableLambda(self._reply)
        self.chain = RunnableWithMessageHistory(
            runnable,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

    def get_session_history(self, session_id: str) -> InMemoryChatMessageHistory:
        return self.sessions.setdefault(session_id, InMemoryChatMessageHistory())

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        return [
            {"role": "human" if isinstance(message, HumanMessage) else "ai", "content": str(message.content)}
            for message in self.get_session_history(session_id).messages
        ]

    def add_message(self, session_id: str, role: str, content: str) -> None:
        history = self.get_session_history(session_id)
        history.add_message(HumanMessage(content=content) if role == "human" else AIMessage(content=content))

    @staticmethod
    def _reply(inputs: Dict[str, Any]) -> str:
        recent = inputs.get("history", [])
        prior_user_messages = [
            str(item.content) for item in recent if isinstance(item, HumanMessage)
        ][-4:]
        return json.dumps({
            "mock_llm_prior_user_messages": prior_user_messages,
            "current_input": inputs.get("input", ""),
        }, ensure_ascii=False)


def demonstrate_memory() -> Tuple[str, str]:
    memory = SessionMemory()
    cfg = {"configurable": {"session_id": "conversation-1"}}
    multi_turn = ["MEMORY DEMO: continuing conversation", ""]
    for query in ("My refund is pending", "How long will it take?"):
        answer = memory.chain.invoke({"input": query}, config=cfg)
        multi_turn.append(f"User: {query}")
        multi_turn.append(f"MOCK_LLM: {answer}")

    fresh_memory = SessionMemory()
    fresh_cfg = {"configurable": {"session_id": "fresh-conversation"}}
    fresh_query = "How long will it take?"
    fresh_history = fresh_memory.get_history("fresh-conversation")
    fresh_answer = fresh_memory.chain.invoke({"input": fresh_query}, config=fresh_cfg)
    fresh_turn = "\n".join([
        "FRESH SESSION DEMO: no previous conversation state",
        "",
        f"History before turn: {fresh_history}",
        f"User: {fresh_query}",
        f"MOCK_LLM: {fresh_answer}",
        "",
    ])
    return "\n".join(multi_turn) + "\n", fresh_turn


def main() -> None:
    multi_turn, fresh_turn = demonstrate_memory()
    transcript_dir = Path(__file__).resolve().parents[2] / "transcripts"
    transcript_dir.mkdir(parents=True, exist_ok=True)
    (transcript_dir / "memory_demo.txt").write_text(multi_turn, encoding="utf-8")
    (transcript_dir / "memory_fresh_session.txt").write_text(fresh_turn, encoding="utf-8")
    print(multi_turn, end="")
    print("Fresh-session evidence saved to transcripts/memory_fresh_session.txt")


if __name__ == "__main__":
    main()
