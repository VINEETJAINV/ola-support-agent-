"""Exercise the API and save representative request-log evidence."""
import io
import json
import logging
import os
import sys
import uuid
from contextlib import redirect_stdout
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from ola_support_agent import api


class LogCapture(logging.Handler):
    def __init__(self):
        super().__init__(logging.INFO)
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


def main():
    capture = LogCapture()
    api.logger.addHandler(capture)
    document_id = f"api-demo-{uuid.uuid4().hex}"
    document_path = ROOT / "kb" / f"{document_id}.txt"
    output = io.StringIO()
    try:
        with redirect_stdout(output):
            with TestClient(api.app) as client:
                first_response = client.post(
                    "/ask",
                    json={
                        "query": "What is the refund policy? My number is +91 98765 43210.",
                        "session_id": "api-demo-phone",
                    },
                )
                first_response.raise_for_status()

                add_response = client.post(
                    "/add-document",
                    json={
                        "title": document_id,
                        "content": "Temporary API demo policy evidence.",
                    },
                )
                add_response.raise_for_status()

                with client.websocket_connect("/ws/chat") as websocket:
                    websocket.send_json({"query": "How do refunds work?"})
                    first_turn = websocket.receive_json()
                    websocket.send_json(
                        {"query": "What about my number 555 123 4567?"}
                    )
                    second_turn = websocket.receive_json()
                    assert first_turn["turn_number"] == 1
                    assert second_turn["turn_number"] == 2
                    websocket.close(code=1000)

                after_disconnect = client.post(
                    "/ask",
                    json={
                        "query": "What are support hours? Call me at 9876543210.",
                        "session_id": "api-demo-after-disconnect",
                    },
                )
                after_disconnect.raise_for_status()

        disconnect_message = next(
            (line for line in output.getvalue().splitlines() if "WebSocket disconnected:" in line),
            None,
        )
        assert disconnect_message is not None
        assert capture.lines
        parsed_lines = [json.loads(line) for line in capture.lines]
        assert all("trace_id" in line and "duration_ms" in line for line in parsed_lines)
        combined_logs = "\n".join(capture.lines)
        assert "[PHONE_REDACTED]" in combined_logs
        assert "98765 43210" not in combined_logs
        assert "555 123 4567" not in combined_logs
        assert "9876543210" not in combined_logs

        transcript = (
            "API TestClient demo: POST /ask, POST /add-document, two WebSocket turns, "
            "client disconnect, then a successful POST /ask.\n"
            f"Disconnect observed: {disconnect_message}\n"
            "Captured JSONL request logs (trace_id and duration_ms included; "
            "phone values masked):\n"
            + "\n".join(capture.lines)
            + "\n"
        )
        transcript_path = ROOT / "transcripts" / "api_demo.txt"
        transcript_path.write_text(transcript, encoding="utf-8")
        print(transcript, end="")
    finally:
        api.logger.removeHandler(capture)
        if document_path.exists():
            document_path.unlink()
        if api.rag_core is not None:
            for collection in (
                api.rag_core.fixed_collection,
                api.rag_core.sentence_collection,
            ):
                collection.delete(where={"doc_id": document_id})


if __name__ == "__main__":
    main()
