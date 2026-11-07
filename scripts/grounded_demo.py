#!/usr/bin/env python
"""
Grounded generation demo with cache.
"""
import os
import sys
import io
from contextlib import redirect_stdout

os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ola_support_agent.rag_core import RAGCore
from ola_support_agent.grounded_generation import GroundedGenerator


def run_demo():
    os.makedirs("transcripts", exist_ok=True)
    
    rag = RAGCore()
    rag.index_documents()
    
    generator = GroundedGenerator(rag)
    
    print("=" * 60)
    print("GROUNDED GENERATION DEMO")
    print("=" * 60)
    
    queries = [
        "What is the refund policy for defective products?",
        "How long does Ola take to respond to P1 tickets?",
        "What happens on the third repeat complaint?",
        "What are the business hours for support?",
        "What compensation do drivers get for P1 outages?",
        "What is the capital of France?"  # Out of scope
    ]
    
    for q in queries:
        resp = generator.generate(q)
        print(f"\nQ: {q}")
        print(f"A: {resp.answer[:200]}...")
        print(f"Confidence: {resp.confidence:.4f}, Fallback: {resp.fallback_triggered}, Cache hit: {resp.cache_hit}")
        print(f"Citations: {len(resp.citations)}")
    
    # Test an identical query twice to demonstrate a miss followed by a hit.
    print("\n--- Cache Test ---")
    cache_query = "How can customers contact support?"
    resp1 = generator.generate(cache_query)
    print(f"First call - Cache hit: {resp1.cache_hit}")
    resp2 = generator.generate(cache_query)
    print(f"Second call - Cache hit: {resp2.cache_hit}")
    
    stats = generator.get_cache_stats()
    print(f"\nCache stats: {stats}")


def main():
    output = io.StringIO()
    with redirect_stdout(output):
        run_demo()
    transcript = output.getvalue()
    print(transcript, end="")
    with open("transcripts/grounded_demo.txt", "w", encoding="utf-8") as f:
        f.write(transcript)
    print("Transcript saved to transcripts/grounded_demo.txt")


if __name__ == "__main__":
    main()