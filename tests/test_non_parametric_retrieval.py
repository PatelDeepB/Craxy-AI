"""
Unit tests for on-disk non-parametric memory retrieval and chunked prefill.

Tests boundary conditions, latency, memory ceilings, and recall accuracy.
"""

import shutil
import tempfile
import numpy as np
import pytest
from experiments.exp003_non_parametric_retrieval import (
    OnDiskVectorIndex,
    RetrievalConfig,
    evaluate_retrieval_performance,
    simulate_chunked_prefill_activation,
)


@pytest.fixture
def temp_index_dir():
    """Create a temporary directory for on-disk index testing."""
    test_dir = tempfile.mkdtemp(prefix="ultron_unit_test_index_")
    yield test_dir
    shutil.rmtree(test_dir, ignore_errors=True)


def test_on_disk_index_build_and_recall(temp_index_dir: str) -> None:
    """Should accurately build memory-mapped vector index and retrieve target ground truth."""
    # Arrange
    config = RetrievalConfig(num_documents=500, embedding_dim=64, top_k=3)
    index = OnDiskVectorIndex(temp_index_dir, config)
    target_id = 42
    target_text = "CRITICAL FACT: On-disk vector index enables factual recall without large parameter overhead."

    # Act
    index.build_index(ground_truth_id=target_id, ground_truth_text=target_text)
    target_query = np.ones(config.embedding_dim, dtype=np.float32)
    top_indices, top_scores, retrieved_texts = index.query_top_k(target_query, top_k=3)
    index.close()

    # Assert
    assert top_indices[0] == target_id
    assert target_text in retrieved_texts
    assert top_scores[0] > 0.95


def test_retrieval_latency_meets_charter_threshold(temp_index_dir: str) -> None:
    """Should verify that average query latency is under the 50 ms charter threshold."""
    # Arrange
    config = RetrievalConfig(num_documents=1000, embedding_dim=128, top_k=3)
    index = OnDiskVectorIndex(temp_index_dir, config)
    index.build_index(ground_truth_id=10, ground_truth_text="Sample fact text.")

    # Act
    metrics = evaluate_retrieval_performance(index, config, ground_truth_id=10, num_benchmark_queries=20)
    index.close()

    # Assert
    assert metrics["avg_query_latency_ms"] < 50.0
    assert metrics["p95_query_latency_ms"] < 50.0


def test_retrieval_ram_delta_stays_under_budget(temp_index_dir: str) -> None:
    """Should ensure memory-mapped querying does not spike process RAM."""
    # Arrange
    config = RetrievalConfig(num_documents=1000, embedding_dim=128, top_k=3)
    index = OnDiskVectorIndex(temp_index_dir, config)
    index.build_index(ground_truth_id=5, ground_truth_text="Sample text.")

    # Act
    metrics = evaluate_retrieval_performance(index, config, ground_truth_id=5, num_benchmark_queries=10)
    index.close()

    # Assert: RAM delta during querying must be minimal (< 50 MB)
    assert metrics["rss_delta_mb"] < 50.0


def test_chunked_prefill_activation_fits_under_ceiling() -> None:
    """Should confirm chunked prefill caps activation memory delta per ADR-004."""
    # Arrange & Act
    result = simulate_chunked_prefill_activation(
        retrieved_context_tokens=512,
        chunk_limit=512,
        hidden_dim=2048
    )

    # Assert
    assert result["fits_under_ceiling"] == 1.0
    assert result["peak_rss_mb"] <= 2048.0
    assert result["activation_delta_mb"] < 150.0
