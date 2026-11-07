#!/usr/bin/env python
"""
Calibration script for grounded generation threshold.
Measures top-1 cosine similarity for in-scope and out-of-scope queries
and saves the threshold to config/threshold.json.
"""
import os
import json
import sys

os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ola_support_agent.rag_core import RAGCore


def calibrate():
    """Run calibration and save threshold."""
    os.makedirs("config", exist_ok=True)
    os.makedirs("transcripts", exist_ok=True)
    
    rag = RAGCore()
    
    # Index documents first
    rag.index_documents()
    
    # In-scope queries (should have high similarity)
    in_scope_queries = [
        "What is the refund policy for defective products?",
        "How long does Ola take to respond to P1 tickets?",
        "What happens on the third repeat complaint?",
        "What are the business hours for support?",
        "What compensation do drivers get for P1 outages?",
    ]
    
    # Out-of-scope queries (should have low similarity)
    out_of_scope_queries = [
        "What is the capital of France?",
        "How do I cook pasta?",
    ]
    
    print("=" * 60)
    print("CALIBRATION RESULTS")
    print("=" * 60)
    
    in_scope_scores = []
    print("\nIn-scope queries:")
    for q in in_scope_queries:
        results = rag.retrieve(q, collection_name="sentence", top_k=1)
        score = results[0]["score"] if results else 0.0
        in_scope_scores.append(score)
        print(f"  '{q[:50]}...' -> top-1 score: {score:.4f}")
    
    out_of_scope_scores = []
    print("\nOut-of-scope queries:")
    for q in out_of_scope_queries:
        results = rag.retrieve(q, collection_name="sentence", top_k=1)
        score = results[0]["score"] if results else 0.0
        out_of_scope_scores.append(score)
        print(f"  '{q[:50]}...' -> top-1 score: {score:.4f}")
    
    in_scope_avg = sum(in_scope_scores) / len(in_scope_scores)
    out_of_scope_avg = sum(out_of_scope_scores) / len(out_of_scope_scores)
    nearest_in_scope = min(in_scope_scores)
    nearest_out_of_scope = max(out_of_scope_scores)
    if nearest_out_of_scope >= nearest_in_scope:
        raise ValueError("Calibration samples overlap; replace them with separable in-scope and out-of-scope examples")
    threshold = (nearest_in_scope + nearest_out_of_scope) / 2
    
    print(f"\nIn-scope average: {in_scope_avg:.4f}")
    print(f"Out-of-scope average: {out_of_scope_avg:.4f}")
    print(f"Chosen threshold (midpoint): {threshold:.4f}")
    
    # Save threshold
    config = {
        "threshold": threshold,
        "in_scope_scores": in_scope_scores,
        "out_of_scope_scores": out_of_scope_scores,
        "in_scope_avg": in_scope_avg,
        "out_of_scope_avg": out_of_scope_avg,
        "in_scope_queries": in_scope_queries,
        "out_of_scope_queries": out_of_scope_queries,
    }
    
    with open("config/threshold.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\nThreshold saved to config/threshold.json")
    
    return threshold


if __name__ == "__main__":
    os.makedirs("transcripts", exist_ok=True)
    os.makedirs("config", exist_ok=True)
    
    threshold = calibrate()
    
    # Save transcript
    with open("transcripts/calibration.txt", "w", encoding="utf-8") as f:
        with open("config/threshold.json", "r", encoding="utf-8") as config_file:
            measured = json.load(config_file)
        f.write("CALIBRATION TRANSCRIPT\n")
        f.write("=" * 60 + "\n")
        f.write("Top-1 cosine scores by query:\n")
        for query, score in zip(measured["in_scope_queries"], measured["in_scope_scores"]):
            f.write(f"IN  {query} => {score:.4f}\n")
        for query, score in zip(measured["out_of_scope_queries"], measured["out_of_scope_scores"]):
            f.write(f"OUT {query} => {score:.4f}\n")
        f.write(f"In-scope mean: {measured['in_scope_avg']:.4f}\n")
        f.write(f"Out-of-scope mean: {measured['out_of_scope_avg']:.4f}\n")
        f.write(
            "Midpoint between nearest observed scores: "
            f"{measured['threshold']:.4f}\n"
        )
