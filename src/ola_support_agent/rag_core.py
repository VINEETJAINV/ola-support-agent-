"""
RAG Core module for Ola Support Agent.
Implements two chunking strategies, embedding, and ChromaDB indexing.
"""
import os
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"
# Prevent Hugging Face/Transformers from attempting even optional metadata calls.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from sentence_transformers import SentenceTransformer
import chromadb


@dataclass
class Chunk:
    text: str
    doc_id: str
    chunk_id: str
    metadata: Dict[str, Any]


class FixedSizeChunker:
    """Fixed-size chunking with overlap."""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        chunks = []
        start = 0
        chunk_idx = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]
            chunk_id = f"{doc_id}_fixed_{chunk_idx}"
            chunks.append(Chunk(
                text=chunk_text,
                doc_id=doc_id,
                chunk_id=chunk_id,
                metadata={"chunk_type": "fixed", "start": start, "end": end}
            ))
            start += self.chunk_size - self.overlap
            chunk_idx += 1
        return chunks


class SentenceChunker:
    """Sentence-based chunking."""
    
    def __init__(self):
        self.sentence_splitter = re.compile(r'(?<=[.!?])\s+')
    
    def chunk(self, text: str, doc_id: str) -> List[Chunk]:
        sentences = self.sentence_splitter.split(text.strip())
        chunks = []
        for idx, sent in enumerate(sentences):
            if sent.strip():
                chunk_id = f"{doc_id}_sent_{idx}"
                chunks.append(Chunk(
                    text=sent.strip(),
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    metadata={"chunk_type": "sentence", "sentence_index": idx}
                ))
        return chunks


class RAGCore:
    """RAG core with dual chunking strategies and ChromaDB indexing."""
    
    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        model_name: str = "all-MiniLM-L6-v2",
        model_cache_dir: Optional[str] = None
    ):
        self.persist_dir = persist_dir
        
        # Load embedding model from local cache
        if model_cache_dir:
            self.model = SentenceTransformer(model_name, cache_folder=model_cache_dir)
        else:
            self.model = SentenceTransformer(model_name)
        
        self.client = chromadb.PersistentClient(path=persist_dir)
        
        # Create collections with cosine similarity
        self.fixed_collection = self.client.get_or_create_collection(
            name="kb_fixed_chunks",
            metadata={"hnsw:space": "cosine"}
        )
        self.sentence_collection = self.client.get_or_create_collection(
            name="kb_sentence_chunks",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.fixed_chunker = FixedSizeChunker()
        self.sentence_chunker = SentenceChunker()
    
    def load_knowledge_base(self, kb_dir: str) -> Dict[str, str]:
        """Load knowledge base documents from directory."""
        docs = {}
        for file_path in Path(kb_dir).glob("*.txt"):
            if file_path.name != "__init__.txt":
                with open(file_path, "r", encoding="utf-8") as f:
                    docs[file_path.stem] = f.read()
        return docs
    
    def index_documents(self, kb_dir: str = "./kb") -> Dict[str, int]:
        """Index documents using both chunking strategies."""
        docs = self.load_knowledge_base(kb_dir)
        fixed_count = 0
        sentence_count = 0
        
        for doc_id, text in docs.items():
            fixed_chunks = self.fixed_chunker.chunk(text, doc_id)
            sentence_chunks = self.sentence_chunker.chunk(text, doc_id)
            
            if fixed_chunks:
                embeddings = self.model.encode([c.text for c in fixed_chunks]).tolist()
                self.fixed_collection.upsert(
                    ids=[c.chunk_id for c in fixed_chunks],
                    embeddings=embeddings,
                    documents=[c.text for c in fixed_chunks],
                    metadatas=[{"doc_id": c.doc_id, **c.metadata} for c in fixed_chunks]
                )
                fixed_count += len(fixed_chunks)
            
            if sentence_chunks:
                embeddings = self.model.encode([c.text for c in sentence_chunks]).tolist()
                self.sentence_collection.upsert(
                    ids=[c.chunk_id for c in sentence_chunks],
                    embeddings=embeddings,
                    documents=[c.text for c in sentence_chunks],
                    metadatas=[{"doc_id": c.doc_id, **c.metadata} for c in sentence_chunks]
                )
                sentence_count += len(sentence_chunks)
        
        return {"fixed_chunks": fixed_count, "sentence_chunks": sentence_count}
    
    def retrieve(
        self,
        query: str,
        collection_name: str = "sentence",
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Retrieve top-k chunks for a query."""
        query_embedding = self.model.encode([query]).tolist()[0]
        collection = self.fixed_collection if collection_name == "fixed" else self.sentence_collection
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        retrieval_results = []
        for i in range(len(results["ids"][0])):
            retrieval_results.append({
                "chunk_id": results["ids"][0][i],
                "doc_id": results["metadatas"][0][i]["doc_id"],
                "text": results["documents"][0][i],
                "score": 1.0 - results["distances"][0][i],  # Convert distance to similarity
                "metadata": results["metadatas"][0][i]
            })
        
        return retrieval_results
    
    def get_collection_stats(self) -> Dict[str, int]:
        return {
            "fixed_count": self.fixed_collection.count(),
            "sentence_count": self.sentence_collection.count()
        }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    
    rag = RAGCore(model_cache_dir="./models")
    stats = rag.index_documents()
    print(f"Indexed: {stats}")
    print(f"Collections: {rag.get_collection_stats()}")
