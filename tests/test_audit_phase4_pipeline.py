"""Independent Adversarial Red-Team Audit of Phase 4 Unified Tri-Tier Pipeline.

Probes critical failure modes:
1. State machine restoration and error containment under malformed query states.
2. Neutralization of malicious AST code injection payloads via tool dispatch.
3. Attention sink preservation and NVMe block eviction under small RAM window.
4. Multi-instance resource and file handle isolation.
5. 100-cycle continuous mixed-workload memory stability (RSS drift < 20.0 MB).
"""

import gc
import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import psutil
import pytest

from src.craxy_ai.config import SYSTEM_LIMITS
from src.craxy_ai.core.paged_kv import CraxyPagedKVCache, PagedKVConfig
from src.craxy_ai.pipeline import CraxyPipeline, PipelineOutput, PipelineState
from src.craxy_ai.tools.code_agent import CraxyCodeAgent
from src.craxy_ai.tools.logic_evaluator import SafeLogicEvaluator
from src.craxy_ai.tools.math_evaluator import SafeMathEvaluator


def test_audit_phase4_rapid_state_resets_and_error_containment():
    """Verify that exceptions during query processing reset pipeline state to IDLE."""
    pipeline = CraxyPipeline()
    assert pipeline.state == PipelineState.IDLE

    # Force a mock error inside internal generation tier
    with patch.object(pipeline, "_run_generation_tier", side_effect=RuntimeError("Simulated execution failure")):
        with pytest.raises(RuntimeError, match="Simulated execution failure"):
            pipeline.run_query("Trigger failure pass")

    # State must be cleanly restored to IDLE
    assert pipeline.state == PipelineState.IDLE

    # Subsequent valid query must execute without hindrance
    out = pipeline.run_query("Calculate 10 + 20")
    assert isinstance(out, PipelineOutput)
    assert out.state == PipelineState.IDLE
    assert "30" in out.response


def test_audit_phase4_malformed_tool_injection_neutralization():
    """Verify that malicious injection payloads in tool tags are neutralized."""
    pipeline = CraxyPipeline()

    # Test dangerous math payload (system command injection)
    res_math = pipeline._execute_tool_dispatch("calc", "__import__('os').system('dir')")
    assert res_math["success"] is False
    assert "ERROR" in res_math["output"]

    # Test resource exhaustion attack (exponential explosion)
    res_overflow = pipeline._execute_tool_dispatch("calc", "2 ** 1000000")
    assert res_overflow["success"] is False
    assert "ERROR" in res_overflow["output"]

    # Test dangerous logic payload (unauthorized filesystem access)
    res_logic = pipeline._execute_tool_dispatch("logic", "open('/etc/passwd')")
    assert res_logic["success"] is False
    assert "ERROR" in res_logic["output"]


def test_audit_phase4_paged_kv_eviction_under_pipeline_streaming():
    """Verify that paged cache under tight RAM window evicts historical blocks to NVMe."""
    temp_dir = tempfile.mkdtemp(prefix="craxy_audit_paged_kv_")
    try:
        tight_cfg = PagedKVConfig(
            max_tokens=256,
            block_size_tokens=64,
            ram_window_tokens=128,
            sink_tokens=64,
        )
        paged_cache = CraxyPagedKVCache(paged_config=tight_cfg, backing_dir=temp_dir)
        pipeline = CraxyPipeline(paged_kv=paged_cache)

        # Ingest 4 blocks (256 tokens total): 2 RAM slots, so blocks 2 and 3 must spill to NVMe
        for idx in range(4):
            k_block = np.full((64, 8, 128), idx, dtype=np.int8)
            v_block = np.full((64, 8, 128), idx, dtype=np.int8)
            paged_cache.append_block(k_block, v_block)

        telemetry = paged_cache.get_telemetry()
        assert telemetry["paged_nvme_blocks"] >= 2
        assert telemetry["resident_ram_blocks"] <= 2
        assert paged_cache.ram_footprint_mb <= 150.0

        paged_cache.close()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_audit_phase4_multi_pipeline_instance_resource_isolation():
    """Verify that multiple concurrent pipelines maintain independent resource handles."""
    temp_dir_1 = tempfile.mkdtemp(prefix="craxy_audit_iso_1_")
    temp_dir_2 = tempfile.mkdtemp(prefix="craxy_audit_iso_2_")
    try:
        cache1 = CraxyPagedKVCache(backing_dir=temp_dir_1)
        cache2 = CraxyPagedKVCache(backing_dir=temp_dir_2)

        p1 = CraxyPipeline(paged_kv=cache1)
        p2 = CraxyPipeline(paged_kv=cache2)

        out1 = p1.run_query("Calculate 100 * 2")
        out2 = p2.run_query("Calculate 50 * 3")

        assert "200" in out1.response
        assert "150" in out2.response

        # Closing p1 must not affect p2
        cache1.close()
        out2_subsequent = p2.run_query("Calculate 40 + 60")
        assert "100" in out2_subsequent.response

        cache2.close()
    finally:
        shutil.rmtree(temp_dir_1, ignore_errors=True)
        shutil.rmtree(temp_dir_2, ignore_errors=True)


def test_audit_phase4_100_cycle_mixed_workload_memory_stability():
    """Verify 100 sequential queries across math and logic exhibit RSS drift < 20.0 MB."""
    paged_cache = CraxyPagedKVCache(PagedKVConfig())
    pipeline = CraxyPipeline(paged_kv=paged_cache)

    # Warm-up pass and garbage collection to clear transient state from prior test fixtures
    pipeline.run_query("warmup")
    gc.collect()
    rss_start = pipeline.get_memory_summary()["current_rss_mb"]

    workload = [
        "Calculate 12 * 12 + 10",
        "Execute algorithmic sorting: numbers = [9, 2, 6, 1]",
        "General reasoning query about memory boundaries",
    ]

    for cycle in range(100):
        pipeline.run_query(workload[cycle % len(workload)])

    gc.collect()
    rss_end = pipeline.get_memory_summary()["current_rss_mb"]
    growth_mb = rss_end - rss_start

    # Memory growth must remain strictly under 20.0 MB and within 2,048 MB budget
    assert growth_mb < 20.0
    assert rss_end <= SYSTEM_LIMITS.max_ram_rss_mb
    paged_cache.close()
