"""
Adversarial Red-Team Audit Suite for Phase 2 Stage 6 (DiskANN Vamana Graph Indexing).

Audits and stresses:
1. Cycle immunity under circular loop graph topologies (A -> B -> C -> A).
2. Disconnected subgraph and isolated zero-degree node handling.
3. Beam width scaling resilience (beam width = 1, 4, 32).
4. Allocator memory stability and zero drift across 1,000 continuous graph queries.
5. Absolute RAM ceiling compliance under sustained graph traversals.
"""

import gc
import os
import shutil
import tempfile
from typing import Generator, Tuple
import numpy as np
import pytest

from src.craxy_ai.config import SYSTEM_LIMITS
from src.craxy_ai.memory.graph_index import VamanaGraphStore, GraphSearchResult


@pytest.fixture
def circular_graph_store() -> Generator[Tuple[VamanaGraphStore, str], None, None]:
    """Create a graph with pure circular cycles to stress cycle immunity."""
    temp_dir = tempfile.mkdtemp(prefix="vamana_cycle_test_")
    graph_path = os.path.join(temp_dir, "cycle_graph.bin")
    store = VamanaGraphStore(storage_path=graph_path, embedding_dim=32, max_degree=4)

    # 4 nodes forming a closed ring: 0 -> 1 -> 2 -> 3 -> 0
    for i in range(4):
        vec = np.zeros(32, dtype=np.float32)
        vec[i] = 1.0
        next_node = (i + 1) % 4
        store.add_node(node_id=i, vector=vec, neighbors=[next_node])

    store.build_disk_graph()
    yield store, graph_path
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_audit_graph_cycle_immunity_and_termination(circular_graph_store):
    """Verify that circular edge topologies terminate without infinite looping."""
    # Arrange
    store, _ = circular_graph_store
    query_vec = np.zeros(32, dtype=np.float32)
    query_vec[2] = 1.0  # target is node 2

    # Act
    result: GraphSearchResult = store.search(query_vector=query_vec, top_k=2, max_hops=40)

    # Assert
    assert result.hops_traversed <= 40
    assert result.nodes_evaluated <= 4
    assert len(result.nearest_nodes) > 0
    top_node_id, top_score = result.nearest_nodes[0]
    assert top_node_id == 2
    assert top_score > 0.99


def test_audit_disconnected_node_handling():
    """Verify searching disconnected nodes does not raise index errors or crash."""
    # Arrange
    temp_dir = tempfile.mkdtemp(prefix="vamana_disconnect_")
    graph_path = os.path.join(temp_dir, "disconnect.bin")
    store = VamanaGraphStore(storage_path=graph_path, embedding_dim=16, max_degree=4)

    # Node 0 has no neighbors (isolated)
    vec0 = np.zeros(16, dtype=np.float32)
    vec0[0] = 1.0
    store.add_node(node_id=0, vector=vec0, neighbors=[])

    # Node 1 is disconnected
    vec1 = np.zeros(16, dtype=np.float32)
    vec1[1] = 1.0
    store.add_node(node_id=1, vector=vec1, neighbors=[])

    store.build_disk_graph()

    # Act: Search target vector matching disconnected node 1
    query_vec = np.zeros(16, dtype=np.float32)
    query_vec[1] = 1.0
    result: GraphSearchResult = store.search(query_vector=query_vec, top_k=1)

    # Assert: Should evaluate entry node 0, find no neighbors, terminate gracefully
    assert result.hops_traversed == 1
    assert result.nodes_evaluated == 1
    assert len(result.nearest_nodes) == 1

    shutil.rmtree(temp_dir, ignore_errors=True)


def test_audit_beam_width_scaling_boundaries(circular_graph_store):
    """Verify beam width scaling across 1, 4, and 32 executes with sub-5ms latency."""
    # Arrange
    store, _ = circular_graph_store
    query_vec = np.zeros(32, dtype=np.float32)
    query_vec[1] = 1.0

    # Act & Assert across varying beam widths
    for beam_w in [1, 4, 32]:
        result = store.search(query_vector=query_vec, top_k=2, beam_width=beam_w)
        assert result.traversal_latency_ms < 5.0
        assert len(result.nearest_nodes) > 0


def test_audit_sustained_1000_graph_searches_memory_invariance(circular_graph_store):
    """Verify 1,000 consecutive disk graph searches produce zero memory leakage."""
    # Arrange
    store, _ = circular_graph_store
    import psutil
    gc.collect()
    rss_initial = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

    query_vec = np.zeros(32, dtype=np.float32)
    query_vec[3] = 1.0

    # Act: 1,000 consecutive searches
    for _ in range(1000):
        res = store.search(query_vector=query_vec, top_k=2)
        assert res.hops_traversed > 0

    gc.collect()
    rss_final = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    memory_drift = rss_final - rss_initial

    # Assert: zero memory drift (< 10 MB) and budget compliance
    assert memory_drift < 10.0
    assert rss_final <= SYSTEM_LIMITS.max_ram_rss_mb
