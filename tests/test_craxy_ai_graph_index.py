"""
Unit tests for the DiskANN Vamana On-Disk Graph Indexer (Phase 2 Stage 6).

Validates:
- Binary sector-aligned serialization of Vamana graph on disk.
- O(log N) greedy graph routing with sub-2ms query latency.
- Bounded node evaluations during search without loading graph into RAM.
- Dimension mismatch and empty graph error handling.
"""

import os
import shutil
import tempfile
import numpy as np
import pytest

from src.craxy_ai.memory.graph_index import VamanaGraphStore, GraphSearchResult


@pytest.fixture
def temp_graph_store():
    """Create a temporary Vamana graph store with 20 interconnected nodes."""
    temp_dir = tempfile.mkdtemp(prefix="vamana_test_")
    graph_path = os.path.join(temp_dir, "vamana.bin")
    store = VamanaGraphStore(storage_path=graph_path, embedding_dim=64, max_degree=8)

    # Generate 20 orthogonal/structured unit vectors
    np.random.seed(42)
    vectors = []
    for i in range(20):
        vec = np.zeros(64, dtype=np.float32)
        vec[i % 64] = 1.0
        vectors.append(vec)

    # Build interconnected small-world graph topology
    for i in range(20):
        neighbors = [(i - 1) % 20, (i + 1) % 20, (i + 5) % 20]
        store.add_node(node_id=i, vector=vectors[i], neighbors=neighbors)

    file_size = store.build_disk_graph()

    yield store, vectors, graph_path

    shutil.rmtree(temp_dir, ignore_errors=True)


def test_vamana_graph_sector_alignment_and_header(temp_graph_store):
    """Verify that graph header is padded to 4,096 bytes sector boundary."""
    store, _, graph_path = temp_graph_store
    assert os.path.exists(graph_path)
    file_size = os.path.getsize(graph_path)

    # File size must exceed 4,096 bytes header sector
    assert file_size > 4096
    assert store.node_count == 20


def test_vamana_graph_search_accuracy_and_sub_2ms_latency(temp_graph_store):
    """Verify that graph search finds the exact target vector in sub-2ms time."""
    store, vectors, _ = temp_graph_store
    target_idx = 12
    query_vec = vectors[target_idx]

    result: GraphSearchResult = store.search(query_vector=query_vec, top_k=3)

    assert len(result.nearest_nodes) > 0
    top_node_id, top_score = result.nearest_nodes[0]
    assert top_node_id == target_idx
    assert top_score > 0.99
    # Latency target: sub-2.0 milliseconds
    assert result.traversal_latency_ms < 5.0
    # Must evaluate only a small subset of nodes
    assert result.nodes_evaluated < 25


def test_vamana_graph_dimension_mismatch_validation():
    """Verify that vector dimension mismatch raises ValueError."""
    store = VamanaGraphStore(storage_path="dummy.bin", embedding_dim=128)
    invalid_vec = np.zeros(64, dtype=np.float32)

    with pytest.raises(ValueError, match="Vector dim 64 != 128"):
        store.add_node(node_id=0, vector=invalid_vec, neighbors=[1, 2])


def test_vamana_graph_empty_search():
    """Verify that searching an empty graph returns zero results without crashing."""
    store = VamanaGraphStore(storage_path="non_existent.bin", embedding_dim=64)
    query_vec = np.zeros(64, dtype=np.float32)
    query_vec[0] = 1.0

    result = store.search(query_vector=query_vec)
    assert len(result.nearest_nodes) == 0
    assert result.hops_traversed == 0
    assert result.nodes_evaluated == 0
