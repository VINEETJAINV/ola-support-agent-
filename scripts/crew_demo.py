"""Demonstrate the CrewAI retrieval and ticket-lookup tools."""
import asyncio
import io
import json
import os
from contextlib import redirect_stdout

os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

from ola_support_agent.crew_runner import run_crew_query


async def run_demo() -> None:
    policy_result = await run_crew_query("What are the business hours for support?")
    ticket_result = await run_crew_query("Check ticket TKT-0001 status", "TKT-0001")

    print("RAG tool query: What are the business hours for support?")
    print(policy_result["answer"])
    print("\nTicket lookup query: Check ticket TKT-0001 status")
    print(json.dumps(ticket_result["ticket_info"], indent=2))
    print("\nCrew response validates:", bool(ticket_result["response_format"]["answer"]))


def main() -> None:
    os.makedirs("transcripts", exist_ok=True)
    output = io.StringIO()
    with redirect_stdout(output):
        asyncio.run(run_demo())
    transcript = output.getvalue()
    print(transcript, end="")
    with open("transcripts/crew_demo.txt", "w", encoding="utf-8") as file:
        file.write(transcript)
    print("Transcript saved to transcripts/crew_demo.txt")


if __name__ == "__main__":
    main()
