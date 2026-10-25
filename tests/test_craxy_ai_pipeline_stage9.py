"""
Unit and integration tests for Phase 2 Stage 9: Pipeline Logic Tool Dispatch & Benchmark.
"""

import gc
import os
import psutil
import pytest

from src.craxy_ai.benchmarks.runner import BenchmarkSuiteRunner
from src.craxy_ai.config import SYSTEM_LIMITS
from src.craxy_ai.pipeline import CraxyPipeline


def test_pipeline_logic_tool_dispatch():
    """Verify that CraxyPipeline dispatches algorithmic logic expressions safely."""
    pipeline = CraxyPipeline()
    res = pipeline._execute_tool_dispatch("logic", "x = [5, 2, 8, 1]; sorted(x)")
    assert res["success"]
    assert res["tool"] == "logic"
    assert res["output"] == "[1, 2, 5, 8]"
    assert res["latency_us"] > 0.0
    assert res["steps"] > 0


def test_pipeline_logic_error_containment():
    """Verify that security violations in tool payloads are safely captured."""
    pipeline = CraxyPipeline()
    res = pipeline._execute_tool_dispatch("logic", "import os")
    assert not res["success"]
    assert "SecurityViolationError" in res["output"]


def test_pipeline_end_to_end_algorithmic_query():
    """Verify end-to-end algorithmic query execution through pipeline."""
    pipeline = CraxyPipeline()
    output = pipeline.run_query("Execute algorithmic sorting: numbers = [9, 4, 1, 7]")
    assert len(output.tool_calls) == 1
    assert output.tool_calls[0]["tool"] == "logic"
    assert "[1, 4, 7, 9]" in output.response
    assert output.peak_rss_mb < SYSTEM_LIMITS.max_ram_rss_mb


def test_humaneval_benchmark_stage9():
    """Verify that BenchmarkSuiteRunner achieves 100% on algorithmic HumanEval tasks."""
    pipeline = CraxyPipeline()
    runner = BenchmarkSuiteRunner(pipeline=pipeline)
    reports = runner.run_all()

    assert reports["HumanEval"].accuracy_pct == 100.0
    assert reports["HumanEval"].total_tasks == 3
    assert reports["HumanEval_Algorithmic"].accuracy_pct == 100.0
    assert reports["HumanEval_Algorithmic"].passed_tasks == 1
    assert reports["HumanEval_Algorithmic"].fits_in_2gb_budget

    assert reports["GSM8K"].accuracy_pct == 100.0
    assert reports["GSM8K"].fits_in_2gb_budget


def test_pipeline_memory_stability_stage9():
    """Verify memory stability across 50 mixed queries."""
    pipeline = CraxyPipeline()
    queries = [
        "Calculate 15 * 4 + 2",
        "Execute algorithmic sorting: numbers = [3, 1, 2]",
        "General reasoning about systems",
    ]

    gc.collect()
    rss_initial = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

    for i in range(50):
        pipeline.run_query(queries[i % len(queries)])

    gc.collect()
    rss_final = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    drift = rss_final - rss_initial

    assert drift < 25.0
    assert rss_final < SYSTEM_LIMITS.max_ram_rss_mb
