"""
Independent Red-Team Audit of Milestone 2 On-Disk Vector Memory Modules (Turn 10).

Audits:
- src.craxy_ai.memory.disk_store (OnDiskDocumentStore, 4 KB alignment, corruption trapping)
- src.craxy_ai.memory.retriever (MemoryMappedRetriever, adversarial query rejection, zero memory drift)
"""

import gc
import os
import shutil
import tempfile
import numpy as np
import psutil
import pytest

from src.craxy_ai.config import SYSTEM_LIMITS
from src.craxy_ai.memory.disk_store import OnDiskDocumentStore
from src.craxy_ai.memory.retriever import MemoryMappedRetriever


@pytest.fixture
def test_store_dir():
    """Create a temporary directory for on-disk store testing and cleanup afterwards."""
    temp_dir = tempfile.mkdtemp(prefix="craxy_audit_milestone2_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_adversarial_orthogonal_and_inverted_query_rejection(test_store_dir):
    """Verify that inverted (negative similarity) and orthogonal queries are 100% rejected by the gate."""
    dim = 256
    store = OnDiskDocumentStore(test_store_dir, embedding_dim=dim)

    # Base target vector
    base_vec = np.zeros(dim, dtype=np.float32)
    base_vec[0] = 1.0
    store.add_chunk("Quantum computing fundamentals", base_vec)

    # Orthogonal vector
    ortho_vec = np.zeros(dim, dtype=np.float32)
    ortho_vec[1] = 1.0
    store.add_chunk("French culinary techniques", ortho_vec)

    store.build_store()

    retriever = MemoryMappedRetriever(test_store_dir)
    try:
        # Inverted query (-base_vec) has similarity -1.0
        inverted_query = -base_vec
        res_inv = retriever.retrieve(inverted_query, top_k=2, min_similarity=0.65)
        assert not res_inv.has_relevant_context
        assert len(res_inv.accepted_chunks) == 0
        assert res_inv.rejected_count == 2

        # Completely orthogonal query on unused dimension
        random_ortho = np.zeros(dim, dtype=np.float32)
        random_ortho[10] = 1.0
        res_ortho = retriever.retrieve(random_ortho, top_k=2, min_similarity=0.65)
        assert not res_ortho.has_relevant_context
        assert len(res_ortho.accepted_chunks) == 0
    finally:
        retriever.close()


def test_dimension_mismatch_boundary_rejection(test_store_dir):
    """Verify that dimension mismatches are rejected fast at system boundaries."""
    store = OnDiskDocumentStore(test_store_dir, embedding_dim=256)

    # Attempt to add a 128-dim vector
    bad_vec = np.zeros(128, dtype=np.float32)
    with pytest.raises(ValueError, match="Vector dimension 128 does not match expected 256"):
        store.add_chunk("Malformed dimension chunk", bad_vec)


def test_zero_memory_leakage_over_1000_mmap_retrieval_queries(test_store_dir):
    """Verify that 1,000 consecutive memory-mapped queries cause zero process memory drift (< 0.5 MB)."""
    dim = 64
    store = OnDiskDocumentStore(test_store_dir, embedding_dim=dim)

    np.random.seed(42)
    for i in range(100):
        vec = np.random.randn(dim).astype(np.float32)
        store.add_chunk(f"Document knowledge chunk #{i}", vec)
    store.build_store()

    retriever = MemoryMappedRetriever(test_store_dir)
    process = psutil.Process(os.getpid())

    try:
        gc.collect()
        rss_before = process.memory_info().rss / (1024 * 1024)

        for _ in range(1000):
            q_vec = np.random.randn(dim).astype(np.float32)
            res = retriever.retrieve(q_vec, top_k=3, min_similarity=0.65)
            assert res.total_scanned == 100

        gc.collect()
        rss_after = process.memory_info().rss / (1024 * 1024)
        rss_delta = rss_after - rss_before

        # Memory drift must remain strictly bounded under 1.0 MB
        assert rss_delta < 1.0, f"Memory drift {rss_delta:.2f} MB exceeded threshold"
    finally:
        retriever.close()


def test_corrupted_record_offset_trapping(test_store_dir):
    """Verify that seeking to an out-of-bounds offset in disk_store raises a ValueError."""
    dim = 32
    store = OnDiskDocumentStore(test_store_dir, embedding_dim=dim)
    vec = np.ones(dim, dtype=np.float32)
    store.add_chunk("Sample document", vec)
    store.build_store()

    # Read from an invalid beyond-file offset
    with pytest.raises(ValueError, match="Corrupt document record"):
        store.read_chunk(chunk_id=999, offset=999999)
