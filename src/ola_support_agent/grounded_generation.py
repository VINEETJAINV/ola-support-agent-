"""
Grounded generation with in-memory cache and calibrated threshold.
"""
import os
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

from ola_support_agent.guardrails import load_threshold
from ola_support_agent.rag_core import RAGCore, Chunk


_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "base", "be", "by", "customer",
    "do", "does", "for", "from", "how", "i", "in", "information", "is", "it",
    "knowledge", "me", "of", "on", "or", "policy", "relevant", "support",
    "the", "ticket", "to", "was", "what", "when", "where", "which", "who",
    "why", "with", "evidence",
}


def _tokens(text: str) -> set[str]:
    """Return lightweight normalized content words for evidence matching."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    normalized = set()
    for word in words:
        if word == "response":
            word = "respond"
        elif word.endswith("ies") and len(word) > 4:
            word = word[:-3] + "y"
        elif word.endswith("ed") and len(word) > 4:
            word = word[:-2]
        elif word.endswith("ing") and len(word) > 5:
            word = word[:-3]
            if len(word) > 3 and word[-1] == word[-2]:
                word = word[:-1]
        elif word.endswith("s") and len(word) > 3:
            word = word[:-1]
        if word not in _STOP_WORDS and len(word) > 1:
            normalized.add(word)
    return normalized


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
        
        self.threshold = load_threshold() if threshold is None else threshold
        if isinstance(self.threshold, bool) or not 0.0 <= self.threshold <= 1.0:
            raise ValueError("Groundedness threshold must be between 0 and 1")
        
        # In-memory cache
        self._cache: Dict[str, GroundedResponse] = {}
        self._call_count = 0
        self._cache_hits = 0
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for cache key."""
        return re.sub(r'\s+', ' ', query.lower().strip())
    
    def generate(self, query: str, context: str = "") -> GroundedResponse:
        """Generate a grounded response for the query."""
        self._call_count += 1
        
        normalized = self._normalize_query(query)
        normalized_context = self._normalize_query(context)
        cache_key = f"{normalized}\n{normalized_context}"
        
        # Check cache
        if cache_key in self._cache:
            self._cache_hits += 1
            cached = self._cache[cache_key]
            return GroundedResponse(
                answer=cached.answer,
                citations=cached.citations,
                confidence=cached.confidence,
                fallback_triggered=cached.fallback_triggered,
                cache_hit=True
            )
        
        # Retrieve relevant chunks
        retrieval_query = f"{context}\n{query}".strip()
        results = self.rag.retrieve(
            retrieval_query,
            collection_name=self.collection_name, 
            top_k=self.top_k
        )
        
        query_tokens = _tokens(query)
        context_tokens = _tokens(context)
        query_relevant = []
        context_relevant = []
        for result in results:
            evidence_tokens = _tokens(result["text"])
            overlap = query_tokens & evidence_tokens
            if result["score"] >= self.threshold and overlap:
                query_relevant.append((result, len(overlap)))
            elif result["score"] >= self.threshold:
                context_overlap = context_tokens & evidence_tokens
                if context_overlap:
                    context_relevant.append((result, len(context_overlap)))
        # Use prior queries only when the current follow-up has no direct lexical match.
        relevant = query_relevant or context_relevant
        relevant.sort(key=lambda item: (item[1], item[0]["score"]), reverse=True)
        selected = [item[0] for item in relevant[:3]]

        if not selected:
            response = GroundedResponse(
                answer="I don't have enough information to answer this question based on the available knowledge base.",
                citations=[],
                confidence=0.0,
                fallback_triggered=True,
                cache_hit=False
            )
        else:
            # This MOCK_LLM path returns selected evidence, not a synthesized answer.
            citations = []
            for result in selected:
                citations.append({
                    "doc_id": result["doc_id"],
                    "chunk_id": result["chunk_id"],
                    "score": round(result["score"], 4),
                    "text_snippet": result["text"][:200]
                })

            answer = "Relevant knowledge-base evidence: " + " ".join(
                result["text"] for result in selected
            )
            confidence = max(result["score"] for result in selected)
            
            response = GroundedResponse(
                answer=answer,
                citations=citations,
                confidence=round(confidence, 4),
                fallback_triggered=False,
                cache_hit=False
            )
        
        # Cache the response
        self._cache[cache_key] = response
        
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