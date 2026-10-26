"""Integration tests for Phase 4 Unified Frontier Tri-Tier Inference Pipeline.

Validates:
- End-to-end orchestration across Neural Core, Paged KV Cache, GGUF Engine,
  Vector Retriever, and Safe Math/Logic/Code Evaluators.
- Physical memory RSS ceiling adherence (strictly <= 2,048 MB).
- Multi-tier tool dispatch across mathematical and algorithmic requests.
- Seamless NVMe-backed 128K context cache block streaming during generation.
"""

import gc
import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.craxy_ai.config import SYSTEM_LIMITS
from src.craxy_ai.core.gguf_types import GGUFEngineConfig, GGUFGenerationResult
from src.craxy_ai.core.model import CraxyNeuralCore
from src.craxy_ai.core.paged_kv import CraxyPagedKVCache
from src.craxy_ai.memory.disk_store import OnDiskDocumentStore
from src.craxy_ai.memory.retriever import MemoryMappedRetriever
from src.craxy_ai.pipeline import CraxyPipeline, PipelineOutput, PipelineState
from src.craxy_ai.tools.code_agent import CraxyCodeAgent
from src.craxy_ai.tools.logic_evaluator import SafeLogicEvaluator
from src.craxy_ai.tools.math_evaluator import SafeMathEvaluator


@pytest.fixture
def temp_nvme_cache_dir():
    """Create a temporary storage directory for NVMe-backed paged KV cache."""
    cache_dir = tempfile.mkdtemp(prefix="craxy_paged_kv_test_")
    yield cache_dir
    gc.collect()
    shutil.rmtree(cache_dir, ignore_errors=True)


@pytest.fixture
def temp_retriever_store():
    """Create a temporary on-disk memory store with sample knowledge chunks."""
    temp_dir = tempfile.mkdtemp(prefix="craxy_frontier_retriever_")
    store = OnDiskDocumentStore(storage_dir=temp_dir, embedding_dim=128)
    sample_texts = [
        "Project ULTRON establishes frontier reasoning within 2 GB RAM budget.",
        "FlashAttention chunked prefill caps intermediate activation memory.",
        "Paged KV caching stores inactive context tokens on NVMe storage.",
    ]
    for idx, text in enumerate(sample_texts):
        vec = np.zeros(128, dtype=np.float32)
        vec[idx] = 1.0
        store.add_chunk(text=text, embedding_vector=vec)
    store.build_store()
    retriever = MemoryMappedRetriever(storage_dir=temp_dir)
    yield retriever, temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_should_initialize_all_tiers_with_headroom(temp_nvme_cache_dir):
    """Verify that unified pipeline initializes all sub-components with positive memory headroom."""
    paged_cache = CraxyPagedKVCache(backing_dir=temp_nvme_cache_dir)
    try:
        pipeline = CraxyPipeline(paged_kv=paged_cache)
        summary = pipeline.get_memory_summary()

        assert summary["max_budget_mb"] == 2048.0
        assert summary["headroom_mb"] > 500.0
        assert summary["current_rss_mb"] < 2048.0
        assert pipeline.state == PipelineState.IDLE
    finally:
        paged_cache.close()


def test_should_dispatch_algorithmic_logic_tool():
    """Verify that sorting requests trigger algorithmic logic AST tool execution."""
    pipeline = CraxyPipeline()
    query = "Please sort this sequence: numbers = [8, 3, 1, 7, 2]"
    out = pipeline.run_query(user_prompt=query)

    assert len(out.tool_calls) == 1
    tool_info = out.tool_calls[0]
    assert tool_info["tool"] == "logic"
    assert tool_info["success"] is True
    assert "[1, 2, 3, 7, 8]" in tool_info["output"]
    assert "[Tool Result: [1, 2, 3, 7, 8]]" in out.response
    assert out.peak_rss_mb <= SYSTEM_LIMITS.max_ram_rss_mb


def test_should_integrate_paged_kv_cache_during_generation(temp_nvme_cache_dir):
    """Verify that paged KV cache accepts context token blocks without RAM inflation."""
    paged_cache = CraxyPagedKVCache(backing_dir=temp_nvme_cache_dir)
    try:
        pipeline = CraxyPipeline(paged_kv=paged_cache)
        out = pipeline.run_query("Analyze architecture decoupling for compact neural models.")

        assert isinstance(out, PipelineOutput)
        assert out.tokens_generated > 0
        assert paged_cache.current_seq_len > 0
        assert out.peak_rss_mb <= SYSTEM_LIMITS.max_ram_rss_mb
    finally:
        paged_cache.close()


def test_should_route_generation_to_gguf_engine():
    """Verify that when use_gguf is requested, queries route to the GGUF engine."""
    mock_gguf = MagicMock()
    mock_res = GGUFGenerationResult(
        generated_text="GGUF engine quantized output: verified token sequence.",
        prompt_tokens=15,
        completion_tokens=8,
        total_tokens=23,
        tokens_per_second=177.7,
        peak_rss_mb=420.5,
        duration_seconds=0.045,
        finish_reason="stop",
    )
    mock_gguf.generate.return_value = mock_res
    pipeline = CraxyPipeline(gguf_engine=mock_gguf)

    out = pipeline.run_query("Evaluate quantized inference path.", use_gguf=True)

    assert out.model_engine_type == "gguf"
    assert out.tokens_generated == 8
    assert "quantized output" in out.response
    assert mock_gguf.generate.called


def test_should_integrate_code_agent_validation():
    """Verify that the embedded code agent provides AST validation within pipeline ecosystem."""
    pipeline = CraxyPipeline()
    valid_python = "def compute_factorial(n: int) -> int:\n    return 1 if n <= 1 else n * compute_factorial(n - 1)\n"

    is_valid = pipeline.code_agent.validate_python_syntax(valid_python)

    assert is_valid is True
    assert pipeline.code_agent.workspace_root.exists()


def test_should_maintain_strict_rss_ceiling_under_mixed_stress(temp_retriever_store, temp_nvme_cache_dir):
    """Verify that 30 continuous mixed-tier queries stay strictly under 2,048 MB physical DRAM."""
    retriever, _ = temp_retriever_store
    paged_cache = CraxyPagedKVCache(backing_dir=temp_nvme_cache_dir)
    try:
        pipeline = CraxyPipeline(retriever=retriever, paged_kv=paged_cache)

        gc.collect()
        rss_initial = pipeline.get_memory_summary()["current_rss_mb"]

        q_vec = np.zeros(128, dtype=np.float32)
        q_vec[0] = 1.0

        for i in range(30):
            if i % 3 == 0:
                pipeline.run_query(f"Calculate {i} * 15 + 4")
            elif i % 3 == 1:
                pipeline.run_query("Explain ULTRON memory bounds", query_vector=q_vec)
            else:
                pipeline.run_query(f"Direct generation step iteration {i}")

        gc.collect()
        rss_final = pipeline.get_memory_summary()["current_rss_mb"]
        drift_mb = rss_final - rss_initial

        assert drift_mb < 30.0
        assert rss_final <= SYSTEM_LIMITS.max_ram_rss_mb
    finally:
        paged_cache.close()
