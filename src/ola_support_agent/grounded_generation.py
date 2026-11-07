"""
Grounded generation with in-memory cache and calibrated threshold.
"""
import os
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

from ola_support_agent.rag_core import RAGCore, Chunk


@dataclass
class GroundedResponse:
    answer: str
    citations: List[Dict[str, Any]]
    confidence: float
    fallback_triggered: bool
    cache_hit: bool


class GroundedGenerator:
    """Grounded generation with calibrated threshold and in-memory cache."""
    
    def __init__(
        self,
        rag: RAGCore,
        collection_name: str = "sentence",
        threshold: Optional[float] = None,
        top_k: int = 5
    ):
        self.rag = rag
        self.collection_name = collection_name
        self.top_k = top_k
        
        # Load threshold from config
        if threshold is not None:
            self.threshold = threshold
        else:
            try:
                with open("config/threshold.json", "r") as f:
                    config = json.load(f)
                    self.threshold = config.get("threshold", 0.3)
            except FileNotFoundError:
                self.threshold = 0.3
        
        # In-memory cache
        self._cache: Dict[str, GroundedResponse] = {}
        self._call_count = 0
        self._cache_hits = 0
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for cache key."""
        return re.sub(r'\s+', ' ', query.lower().strip())
    
    def generate(self, query: str) -> GroundedResponse:
        """Generate a grounded response for the query."""
        self._call_count += 1
        
        normalized = self._normalize_query(query)
        
        # Check cache
        if normalized in self._cache:
            self._cache_hits += 1
            cached = self._cache[normalized]
            return GroundedResponse(
                answer=cached.answer,
                citations=cached.citations,
                confidence=cached.confidence,
                fallback_triggered=cached.fallback_triggered,
                cache_hit=True
            )
        
        # Retrieve relevant chunks
        results = self.rag.retrieve(
            query, 
            collection_name=self.collection_name, 
            top_k=self.top_k
        )
        
        if not results or results[0]["score"] < self.threshold:
            response = GroundedResponse(
                answer="I don't have enough information to answer this question based on the available knowledge base.",
                citations=[],
                confidence=0.0,
                fallback_triggered=True,
                cache_hit=False
            )
        else:
            # Build answer from retrieved chunks
            answer_parts = []
            citations = []
            
            for r in results:
                if r["score"] >= self.threshold:
                    answer_parts.append(r["text"])
                    citations.append({
                        "doc_id": r["doc_id"],
                        "chunk_id": r["chunk_id"],
                        "score": round(r["score"], 4),
                        "text_snippet": r["text"][:200]
                    })
            
            answer = " ".join(answer_parts) if answer_parts else "I don't have enough information to answer this question."
            confidence = results[0]["score"] if results else 0.0
            
            response = GroundedResponse(
                answer=answer,
                citations=citations,
                confidence=round(confidence, 4),
                fallback_triggered=False,
                cache_hit=False
            )
        
        # Cache the response
        self._cache[normalized] = response
        
        return GroundedResponse(
            answer=response.answer,
            citations=response.citations,
            confidence=response.confidence,
            fallback_triggered=response.fallback_triggered,
            cache_hit=False
        )
    
    def get_cache_stats(self) -> Dict[str, int]:
        return {
            "total_calls": self._call_count,
            "cache_hits": self._cache_hits,
            "cache_misses": self._call_count - self._cache_hits,
            "cache_size": len(self._cache)
        }


# Import json here to avoid circular imports
import json

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    
    from ola_support_agent.rag_core import RAGCore
    
    rag = RAGCore()
    rag.index_documents()
    
    generator = GroundedGenerator(rag)
    
    print("=" * 60)
    print("GROUNDED GENERATION DEMO")
    print("=" * 60)
    
    # Test in-scope queries
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
    
    # Test cache
    print("\n--- Cache Test ---")
    resp1 = generator.generate("What is the refund policy for defective products?")
    print(f"First call - Cache hit: {resp1.cache_hit}")
    resp2 = generator.generate("What is the refund policy for defective products?")
    print(f"Second call - Cache hit: {resp2.cache_hit}")
    
    stats = generator.get_cache_stats()
    print(f"\nCache stats: {stats}")