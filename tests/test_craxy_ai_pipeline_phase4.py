"""Unit and integration tests for Phase 4: Unified Frontier Tri-Tier Pipeline.

Validates end-to-end integration of:
- Decoupled parametric reasoning with CraxyNeuralCore and CraxyGGUFEngine.
- 128K context awareness with CraxyPagedKVCache.
- In-process tool dispatching across SafeMath, SafeLogic, and CraxyCodeAgent.
- Strict physical RAM adherence under repeated multi-modal query cycles.
"""

import gc
import os
import psutil
import pytest

from src.craxy_ai.config import SYSTEM_LIMITS
from src.craxy_ai.core.gguf_engine import CraxyGGUFEngine
from src.craxy_ai.core.model import CraxyNeuralCore
from src.craxy_ai.core.paged_kv import CraxyPagedKVCache, PagedKVConfig
from src.craxy_ai.pipeline import CraxyPipeline, PipelineOutput, PipelineState
from src.craxy_ai.tools.code_agent import CraxyCodeAgent


def test_pipeline_phase4_initialization_with_all_tiers():
    """Verify that CraxyPipeline initializes all Phase 4 tiers with positive headroom."""
    paged_kv = CraxyPagedKVCache(PagedKVConfig())
    gguf_engine = CraxyGGUFEngine()
    code_agent = CraxyCodeAgent()

    pipeline = CraxyPipeline(
        paged_kv=paged_kv,
        gguf_engine=gguf_engine,
        code_agent=code_agent,
    )
    summary = pipeline.get_memory_summary()

    assert summary["max_budget_mb"] == 2048.0
    assert summary["core_int4_mb"] > 700.0
    assert summary["kv_cache_mb"] == 112.0
    assert summary["headroom_mb"] > 500.0
    assert pipeline.state == PipelineState.IDLE
    paged_kv.close()
    gguf_engine.close()


def test_pipeline_phase4_gguf_generation_mode():
    """Verify that use_gguf=True invokes the quantized GGUF neural engine."""
    gguf_engine = CraxyGGUFEngine()
    pipeline = CraxyPipeline(gguf_engine=gguf_engine)

    output = pipeline.run_query("Synthesize executive reasoning summary", use_gguf=True)

    assert isinstance(output, PipelineOutput)
    assert output.model_engine_type == "gguf"
    assert output.tokens_generated > 0
    assert output.peak_rss_mb <= SYSTEM_LIMITS.max_ram_rss_mb
    assert output.state == PipelineState.IDLE
    gguf_engine.close()


def test_pipeline_phase4_paged_kv_block_tracking():
    """Verify that paged KV cache tracks context blocks during generation."""
    paged_kv = CraxyPagedKVCache(PagedKVConfig())
    pipeline = CraxyPipeline(paged_kv=paged_kv)

    output = pipeline.run_query("General system query with long context awareness")

    assert output.tokens_generated > 0
    assert output.context_tokens_used > 0
    assert paged_kv.active_token_count > 0
    assert output.peak_rss_mb <= SYSTEM_LIMITS.max_ram_rss_mb
    paged_kv.close()


def test_pipeline_phase4_multi_tool_mixed_workload():
    """Verify that math and logic tools dispatch correctly across sequential queries."""
    pipeline = CraxyPipeline()

    out_math = pipeline.run_query("Calculate 120 * 5 + 40 for distribution.")
    assert len(out_math.tool_calls) == 1
    assert out_math.tool_calls[0]["tool"] == "calc"
    assert "640" in out_math.response

    out_logic = pipeline.run_query("Execute algorithmic sorting: numbers = [8, 3, 5, 1]")
    assert len(out_logic.tool_calls) == 1
    assert out_logic.tool_calls[0]["tool"] == "logic"
    assert "[1, 3, 5, 8]" in out_logic.response


def test_pipeline_phase4_sustained_ram_stability():
    """Verify that 30 continuous mixed queries exhibit minimal memory drift."""
    paged_kv = CraxyPagedKVCache(PagedKVConfig())
    pipeline = CraxyPipeline(paged_kv=paged_kv)
    gc.collect()
    rss_start = pipeline.get_memory_summary()["current_rss_mb"]

    queries = [
        "Calculate 50 * 4 + 10",
        "Execute algorithmic sorting: numbers = [4, 2, 7]",
        "General reasoning about memory walls",
    ]

    for i in range(30):
        pipeline.run_query(queries[i % len(queries)])

    gc.collect()
    rss_end = pipeline.get_memory_summary()["current_rss_mb"]
    drift_mb = abs(rss_end - rss_start)

    # Drift must remain strictly under 25.0 MB
    assert drift_mb < 25.0
    assert rss_end <= SYSTEM_LIMITS.max_ram_rss_mb
    paged_kv.close()
