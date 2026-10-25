"""
Memory-mapped vector retrieval engine with calibrated relevance gating.

Implements ADR-005 and ADR-006:
- Zero-copy vector search backed by memory-mapped NVMe storage.
- Strict relevance threshold filtering (tau >= 0.65) to eliminate hallucinated context.
- High-throughput top-k selection with byte-offset document dereferencing.
"""

import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.craxy_ai.config import SYSTEM_LIMITS
from src.craxy_ai.memory.disk_store import DocumentChunk


@dataclass(frozen=True)
class ScoredChunk:
    """Document chunk paired with its query cosine similarity score."""
    chunk: DocumentChunk
    similarity_score: float


@dataclass(frozen=True)
class RetrievalResult:
    """Encapsulates the output of a memory-mapped query search."""
    accepted_chunks: List[ScoredChunk]
    rejected_count: int
    total_scanned: int
    search_latency_ms: float
    relevance_threshold: float

    @property
    def has_relevant_context(self) -> bool:
        """Return True if at least one retrieved chunk passed the relevance gate."""
        return len(self.accepted_chunks) > 0


class MemoryMappedRetriever:
    """
    Executes sub-millisecond similarity search against on-disk binary vectors.
    
    Uses memory-mapped file handles to keep host process RAM delta under 15 MB.
    """

    def __init__(self, storage_dir: str) -> None:
        self.storage_dir = storage_dir
        self.vector_file_path = os.path.join(storage_dir, "vectors.bin")
        self.doc_file_path = os.path.join(storage_dir, "documents.bin")
        self.index_file_path = os.path.join(storage_dir, "index.json")

        if not os.path.exists(self.index_file_path):
            raise FileNotFoundError(f"Storage index not found at {self.index_file_path}")

        with open(self.index_file_path, "r", encoding="utf-8") as f_idx:
            manifest = json.load(f_idx)

        self.num_chunks: int = manifest["num_chunks"]
        self.embedding_dim: int = manifest["embedding_dim"]
        self.chunk_offsets: Dict[int, int] = {
            item["chunk_id"]: item["offset"] for item in manifest["chunks"]
        }

        # Memory-map the binary vector array in read-only mode
        self.vectors_mmap = np.memmap(
            self.vector_file_path,
            dtype=np.float32,
            mode="r",
            shape=(self.num_chunks, self.embedding_dim)
        )

    def retrieve(
        self,
        query_vector: np.ndarray,
        top_k: int = 3,
        min_similarity: float = SYSTEM_LIMITS.min_retrieval_similarity_threshold
    ) -> RetrievalResult:
        """
        Execute cosine similarity search and filter results via relevance gate.
        
        Discards any document chunks scoring below min_similarity (ADR-006).
        """
        t0 = time.perf_counter()
        
        # Normalize query vector for cosine similarity
        norm = np.linalg.norm(query_vector)
        if norm < 1e-9:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return RetrievalResult([], 0, self.num_chunks, elapsed_ms, min_similarity)

        q_norm = query_vector / norm
        
        # Fast dot-product against memory-mapped array
        scores = np.dot(self.vectors_mmap, q_norm).flatten()
        
        # Partition top-k candidate indices
        k_bounded = min(top_k, self.num_chunks)
        candidate_indices = np.argpartition(scores, -k_bounded)[-k_bounded:]
        sorted_indices = candidate_indices[np.argsort(-scores[candidate_indices])]

        accepted: List[ScoredChunk] = []
        rejected_count = 0

        # Dereference text for candidate chunks from disk
        with open(self.doc_file_path, "rb") as f_doc:
            for idx in sorted_indices:
                score = float(scores[idx])
                if score >= min_similarity:
                    offset = self.chunk_offsets[int(idx)]
                    f_doc.seek(offset)
                    text_len = np.frombuffer(f_doc.read(4), dtype="<u4")[0]
                    text = f_doc.read(text_len).decode("utf-8")
                    
                    chunk = DocumentChunk(
                        chunk_id=int(idx),
                        text=text,
                        metadata={},
                        byte_offset=offset,
                        byte_length=text_len + 4
                    )
                    accepted.append(ScoredChunk(chunk=chunk, similarity_score=score))
                else:
                    rejected_count += 1

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return RetrievalResult(
            accepted_chunks=accepted,
            rejected_count=rejected_count,
            total_scanned=self.num_chunks,
            search_latency_ms=elapsed_ms,
            relevance_threshold=min_similarity
        )

    def close(self) -> None:
        """Release the memory-mapped file descriptor."""
        if hasattr(self, "vectors_mmap") and self.vectors_mmap is not None:
            del self.vectors_mmap
            self.vectors_mmap = None
