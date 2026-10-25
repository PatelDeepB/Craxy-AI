"""
Adversarial Red-Team Audit Suite for Milestone 4 (Tri-Tier Unified Pipeline).

Audits and stresses:
1. Adversarial input validation and error containment in tool dispatch.
2. Orthogonal and negative query vector rejection under relevance gating.
3. Multi-turn dialogue stability and rolling context token bounding.
4. Process RSS memory ceiling adherence and allocator drift across 50 turns.
5. In-process telemetry integrity across pipeline states.
"""

import gc
import os
import shutil
import tempfile
from typing import Generator, Tuple
import numpy as np
import pytest

from src.craxy_ai.config import SYSTEM_LIMITS
from src.craxy_ai.memory.disk_store import OnDiskDocumentStore
from src.craxy_ai.memory.retriever import MemoryMappedRetriever
from src.craxy_ai.pipeline import CraxyPipeline, PipelineState


@pytest.fixture
def populated_retriever() -> Generator[Tuple[MemoryMappedRetriever, str], None, None]:
    """Provide a validated on-disk retriever with 3 orthogonal unit vectors."""
    temp_dir = tempfile.mkdtemp(prefix="audit_pipe_store_")
    store = OnDiskDocumentStore(storage_dir=temp_dir, embedding_dim=64)

    test_knowledge = [
        "Knowledge Alpha: FlashAttention chunks KV cache sequentially.",
        "Knowledge Beta: On-disk vector indices avoid RAM bloat.",
        "Knowledge Gamma: AST tool sandboxes eliminate arithmetic hallucinations."
    ]

    for index, text in enumerate(test_knowledge):
        vector = np.zeros(64, dtype=np.float32)
        vector[index] = 1.0
        store.add_chunk(text=text, embedding_vector=vector)

    store.build_store()
    retriever = MemoryMappedRetriever(storage_dir=temp_dir)

    yield retriever, temp_dir

    shutil.rmtree(temp_dir, ignore_errors=True)


def test_audit_adversarial_zero_division_tool_handling():
    """Verify pipeline gracefully traps zero division without crashing."""
    # Arrange
    pipeline = CraxyPipeline()
    adversarial_query = "Please compute 100 / 0 immediately."

    # Act
    output = pipeline.run_query(adversarial_query)

    # Assert
    assert len(output.tool_calls) == 1
    tool_telemetry = output.tool_calls[0]
    assert tool_telemetry["tool"] == "calc"
    assert tool_telemetry["success"] is False
    assert "ZeroDivisionError" in tool_telemetry["output"]
    assert "ERROR" in output.response
    assert output.peak_rss_mb <= SYSTEM_LIMITS.max_ram_rss_mb
    assert output.state == PipelineState.IDLE


def test_audit_relevance_gate_rejects_orthogonal_queries(populated_retriever):
    """Verify retriever gate rejects queries below 0.65 similarity."""
    # Arrange
    retriever, _ = populated_retriever
    pipeline = CraxyPipeline(retriever=retriever)
    
    # Query vector orthogonal to all indexed vectors (index 50)
    orthogonal_vector = np.zeros(64, dtype=np.float32)
    orthogonal_vector[50] = 1.0

    # Act
    output = pipeline.run_query("Irrelevant topic query", query_vector=orthogonal_vector)

    # Assert
    assert output.retrieval_augmented is False
    assert output.retrieved_chunks_count == 0
    assert "Knowledge Alpha" not in output.response
    assert output.peak_rss_mb <= SYSTEM_LIMITS.max_ram_rss_mb


def test_audit_relevance_gate_accepts_valid_queries(populated_retriever):
    """Verify retriever gate accepts queries matching indexed knowledge."""
    # Arrange
    retriever, _ = populated_retriever
    pipeline = CraxyPipeline(retriever=retriever)
    
    # Query vector aligning with Knowledge Alpha (index 0)
    matching_vector = np.zeros(64, dtype=np.float32)
    matching_vector[0] = 1.0

    # Act
    output = pipeline.run_query("Explain KV cache chunking", query_vector=matching_vector)

    # Assert
    assert output.retrieval_augmented is True
    assert output.retrieved_chunks_count == 1
    assert "Knowledge Alpha: FlashAttention" in output.response


def test_audit_multi_turn_dialogue_bounds_context_history():
    """Verify rolling context window retains only recent turns to prevent context blowup."""
    # Arrange
    pipeline = CraxyPipeline()
    
    # Act: Run 10 conversation turns
    for turn_idx in range(10):
        pipeline.run_query(f"User message turn {turn_idx}")

    # Assert: History has 10 entries, but context assembly limits prompt to last 3 turns
    assert len(pipeline.conversation_history) == 10
    assembled_prompt, estimated_tokens = pipeline._assemble_context("Final question", [])
    
    assert "User message turn 9" in assembled_prompt
    assert "User message turn 8" in assembled_prompt
    assert "User message turn 7" in assembled_prompt
    assert "User message turn 0" not in assembled_prompt
    assert estimated_tokens < 200


def test_audit_pipeline_extended_stress_under_ram_ceiling():
    """Stress pipeline across 50 alternating queries and verify memory remains <= 2048 MB."""
    # Arrange
    pipeline = CraxyPipeline()
    gc.collect()
    rss_start = pipeline.get_memory_summary()["current_rss_mb"]

    # Act: 50 turns alternating between math calculation and standard reasoning
    for cycle in range(50):
        if cycle % 2 == 0:
            query = f"Please calculate {cycle} * 10 + 5"
        else:
            query = f"Explain reasoning concept {cycle}"
        
        output = pipeline.run_query(query)
        assert output.peak_rss_mb <= SYSTEM_LIMITS.max_ram_rss_mb

    gc.collect()
    rss_end = pipeline.get_memory_summary()["current_rss_mb"]
    memory_drift = rss_end - rss_start

    # Assert allocator stability
    assert memory_drift < 30.0
    assert rss_end <= SYSTEM_LIMITS.max_ram_rss_mb
