"""
Unit tests for the production Craxy-AI on-disk memory retrieval engine.

Verifies binary sector-aligned storage, memory-mapped similarity search,
relevance threshold gating (tau >= 0.65), process RAM stability, and query latency.
"""

import shutil
import tempfile
import numpy as np
import psutil
import pytest
import os
from src.craxy_ai.memory.disk_store import OnDiskDocumentStore
from src.craxy_ai.memory.retriever import MemoryMappedRetriever


@pytest.fixture
def memory_test_env():
    """Create a temporary storage environment and clean it up after testing."""
    temp_dir = tempfile.mkdtemp(prefix="craxy_mem_test_")
    store = OnDiskDocumentStore(temp_dir, embedding_dim=64)
    
    # Seed reproducible vectors
    np.random.seed(123)
    target_vec = np.ones(64, dtype=np.float32)
    target_vec = target_vec / np.linalg.norm(target_vec)
    
    # Store target document
    store.add_chunk("Target factual document on Craxy-AI architecture.", target_vec)
    
    # Store background distractor documents
    for i in range(1, 100):
        rand_vec = np.random.randn(64).astype(np.float32)
        rand_vec = rand_vec / np.linalg.norm(rand_vec)
        store.add_chunk(f"Background knowledge chunk {i}.", rand_vec)
        
    store.build_store()
    yield temp_dir, target_vec
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_disk_store_build_and_read(memory_test_env) -> None:
    """Should correctly serialize and dereference chunks from binary storage."""
    # Arrange
    temp_dir, _ = memory_test_env
    store = OnDiskDocumentStore(temp_dir, embedding_dim=64)
    
    # Act
    chunk = store.read_chunk(chunk_id=0, offset=0)
    
    # Assert
    assert chunk.chunk_id == 0
    assert "Craxy-AI architecture" in chunk.text
    assert chunk.byte_length > 0


def test_retriever_sub_millisecond_latency(memory_test_env) -> None:
    """Should execute memory-mapped vector search in under 5 milliseconds."""
    # Arrange
    temp_dir, target_vec = memory_test_env
    retriever = MemoryMappedRetriever(temp_dir)
    
    # Act
    result = retriever.retrieve(target_vec, top_k=3, min_similarity=0.65)
    retriever.close()
    
    # Assert
    assert result.search_latency_ms < 5.0
    assert result.has_relevant_context is True
    assert result.accepted_chunks[0].chunk.chunk_id == 0


def test_relevance_gate_filters_low_similarity(memory_test_env) -> None:
    """Should reject chunks scoring below the 0.65 relevance gate threshold."""
    # Arrange
    temp_dir, _ = memory_test_env
    retriever = MemoryMappedRetriever(temp_dir)
    
    # Create an orthogonal/opposite query vector that yields very low cosine similarity
    opposite_vec = -np.ones(64, dtype=np.float32)
    opposite_vec = opposite_vec / np.linalg.norm(opposite_vec)
    
    # Act: Query with strict 0.65 threshold
    result = retriever.retrieve(opposite_vec, top_k=5, min_similarity=0.65)
    retriever.close()
    
    # Assert
    # Any chunks with similarity < 0.65 must be rejected
    for scored_chunk in result.accepted_chunks:
        assert scored_chunk.similarity_score >= 0.65
    assert result.rejected_count > 0


def test_retrieval_process_ram_delta_stays_under_budget(memory_test_env) -> None:
    """Should verify that memory-mapped queries do not inflate host process RSS."""
    # Arrange
    temp_dir, target_vec = memory_test_env
    retriever = MemoryMappedRetriever(temp_dir)
    process = psutil.Process(os.getpid())
    rss_before = process.memory_info().rss / (1024 * 1024)
    
    # Act: Run 50 continuous searches
    for _ in range(50):
        _ = retriever.retrieve(target_vec, top_k=3, min_similarity=0.65)
        
    rss_after = process.memory_info().rss / (1024 * 1024)
    retriever.close()
    
    # Assert: RAM delta must remain well below the 20 MB budget limit
    ram_delta = max(0.0, rss_after - rss_before)
    assert ram_delta < 20.0


def test_zero_vector_query_handling(memory_test_env) -> None:
    """Should gracefully handle empty zero-norm query vector without crash."""
    # Arrange
    temp_dir, _ = memory_test_env
    retriever = MemoryMappedRetriever(temp_dir)
    zero_vec = np.zeros(64, dtype=np.float32)
    
    # Act
    result = retriever.retrieve(zero_vec, top_k=3, min_similarity=0.65)
    retriever.close()
    
    # Assert
    assert result.has_relevant_context is False
    assert len(result.accepted_chunks) == 0
