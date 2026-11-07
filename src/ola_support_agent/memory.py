"""Process-local conversation memory backed by LangChain message history."""
from typing import Any, Dict, List

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
        context = " | ".join(str(item.content) for item in recent[-4:]) or "No prior conversation."
        return f"MOCK_LLM context: {context}; current input: {inputs.get('input', '')}"


def demonstrate_memory() -> None:
    memory = SessionMemory()
    cfg = {"configurable": {"session_id": "conversation-1"}}
    for query in ("My refund is pending", "How long will it take?"):
        print(memory.chain.invoke({"input": query}, config=cfg))
    print("fresh session:", memory.get_history("conversation-2"))


if __name__ == "__main__":
    demonstrate_memory()
