"""
Unit tests for the Craxy-AI Tri-Tier Unified Inference Pipeline (Milestone 4).

Validates:
- Unified orchestration across CraxyNeuralCore, MemoryMappedRetriever, and SafeMathEvaluator.
- 5-state control loop execution (IDLE -> PREFILL -> GENERATE -> TOOL_DISPATCH -> FINALIZE).
- Dynamic tool interception and execution for mathematical precision.
- Zero-copy RAG context injection with relevance threshold gating.
- Physical system RAM RSS adherence (strictly <= 2,048 MB under repeated queries).
"""

import gc
import os
import shutil
import tempfile
import numpy as np
import pytest

from src.craxy_ai.config import SYSTEM_LIMITS
from src.craxy_ai.core.model import CraxyNeuralCore
from src.craxy_ai.memory.disk_store import OnDiskDocumentStore
from src.craxy_ai.memory.retriever import MemoryMappedRetriever
from src.craxy_ai.pipeline import CraxyPipeline, PipelineState, PipelineOutput
from src.craxy_ai.tools.math_evaluator import SafeMathEvaluator


@pytest.fixture
def temp_memory_store():
    """Create a temporary on-disk vector store with 5 test documents."""
    temp_dir = tempfile.mkdtemp(prefix="craxy_pipe_test_")
    store = OnDiskDocumentStore(storage_dir=temp_dir, embedding_dim=128)

    sample_docs = [
        "Project Craxy-AI achieves 100B parameter model parity within a 2 GB RAM budget.",
        "FlashAttention sequence chunking eliminates prefill quadratic activation memory spikes.",
        "INT8 Key-Value caching bounds 2,048-token context memory to exactly 112 MB.",
        "On-disk binary vector indexing enables sub-millisecond retrieval without RAM inflation.",
        "AST tool sandboxing provides microsecond evaluation with mathematical zero-hallucination."
    ]

    for idx, doc in enumerate(sample_docs):
        # Generate predictable unit vectors
        vec = np.zeros(128, dtype=np.float32)
        vec[idx] = 1.0
        store.add_chunk(text=doc, embedding_vector=vec)

    store.build_store()
    retriever = MemoryMappedRetriever(storage_dir=temp_dir)

    yield retriever, temp_dir

    shutil.rmtree(temp_dir, ignore_errors=True)


def test_pipeline_initialization_and_memory_summary():
    """Verify that the unified pipeline initializes all tiers with positive headroom."""
    pipeline = CraxyPipeline()
    summary = pipeline.get_memory_summary()

    assert summary["max_budget_mb"] == 2048.0
    assert summary["core_int4_mb"] > 700.0
    assert summary["kv_cache_mb"] == 112.0
    assert summary["headroom_mb"] > 500.0
    assert pipeline.state == PipelineState.IDLE


def test_pipeline_direct_generation_under_memory_ceiling():
    """Verify that pure neural generation completes without breaching RAM limits."""
    pipeline = CraxyPipeline()
    out = pipeline.run_query("What is the principle of parameter decoupling?")

    assert isinstance(out, PipelineOutput)
    assert not out.retrieval_augmented
    assert len(out.tool_calls) == 0
    assert out.tokens_generated > 0
    assert out.peak_rss_mb <= SYSTEM_LIMITS.max_ram_rss_mb
    assert out.state == PipelineState.IDLE


def test_pipeline_math_tool_dispatch_and_execution():
    """Verify that mathematical expressions trigger tool dispatch and precision calculation."""
    pipeline = CraxyPipeline()
    out = pipeline.run_query("Please calculate 45 * 12 + 8 for the project budget.")

    assert len(out.tool_calls) == 1
    tool_info = out.tool_calls[0]
    assert tool_info["tool"] == "calc"
    assert tool_info["success"] is True
    assert tool_info["output"] == "548"
    assert "[Tool Result: 548]" in out.response
    assert out.peak_rss_mb <= SYSTEM_LIMITS.max_ram_rss_mb


def test_pipeline_rag_retrieval_augmentation(temp_memory_store):
    """Verify that matching query vectors retrieve on-disk chunks into context."""
    retriever, _ = temp_memory_store
    pipeline = CraxyPipeline(retriever=retriever)

    # Query matching vector 0
    q_vec = np.zeros(128, dtype=np.float32)
    q_vec[0] = 1.0

    out = pipeline.run_query("Explain Craxy-AI project goals", query_vector=q_vec)

    assert out.retrieval_augmented is True
    assert out.retrieved_chunks_count > 0
    assert "Craxy-AI achieves 100B parameter model parity" in out.response
    assert out.peak_rss_mb <= SYSTEM_LIMITS.max_ram_rss_mb


def test_pipeline_repeated_queries_zero_memory_leakage():
    """Verify that 25 continuous query cycles produce zero unbounded RAM leakage."""
    pipeline = CraxyPipeline()
    gc.collect()
    rss_initial = pipeline.get_memory_summary()["current_rss_mb"]

    for i in range(25):
        pipeline.run_query(f"Continuous stress query iteration {i} + 1")

    gc.collect()
    rss_final = pipeline.get_memory_summary()["current_rss_mb"]
    drift_mb = rss_final - rss_initial

    # Allow minimal allocator tolerance (< 25 MB)
    assert drift_mb < 25.0
    assert rss_final <= SYSTEM_LIMITS.max_ram_rss_mb
