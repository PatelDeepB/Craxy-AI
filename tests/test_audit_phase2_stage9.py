"""
Independent Adversarial Red-Team Audit for Phase 2 Stage 9 Pipeline Tool Orchestration.

Audits:
1. Tool dispatch error containment under corrupted and adversarial payloads (malformed math, unsafe logic, unknown tools).
2. State machine invariance: ensure pipeline.state resets to IDLE under all conditions.
3. Multi-turn conversation history rolling context window bounds.
4. Benchmark verifier adversarial strictness against false positives.
5. Allocator stability (< 15.0 MB drift) over 150 mixed-tool end-to-end pipeline executions.
"""

import gc
import os
import psutil
import pytest

from src.craxy_ai.benchmarks.runner import (
    GSM8KBenchmark,
    HumanEvalBenchmark,
    MMLUBenchmark,
)
from src.craxy_ai.config import SYSTEM_LIMITS
from src.craxy_ai.pipeline import CraxyPipeline, PipelineState


def test_audit_tool_dispatch_corrupted_payloads_containment():
    """Verify that malformed or dangerous tool payloads fail closed safely."""
    pipeline = CraxyPipeline()

    # 1. Corrupted math payload
    res_math_bad = pipeline._execute_tool_dispatch("calc", "**//!!invalid")
    assert not res_math_bad["success"]
    assert "ERROR" in res_math_bad["output"]

    # 2. Division by zero in math tool
    res_div_zero = pipeline._execute_tool_dispatch("calc", "100 / 0")
    assert not res_div_zero["success"]
    assert "ZeroDivisionError" in res_div_zero["output"]

    # 3. Security violation in logic tool
    res_logic_sec = pipeline._execute_tool_dispatch("logic", "import subprocess")
    assert not res_logic_sec["success"]
    assert "SecurityViolationError" in res_logic_sec["output"]

    # 4. Step limit violation in logic tool
    res_logic_step = pipeline._execute_tool_dispatch("logic", "while True: pass")
    assert not res_logic_step["success"]
    assert "StepLimitExceededError" in res_logic_step["output"]

    # 5. Unknown tool request
    res_unknown = pipeline._execute_tool_dispatch("shell", "ls -la")
    assert not res_unknown["success"]
    assert res_unknown["output"] == "UNKNOWN_TOOL"


def test_audit_state_machine_resets_to_idle():
    """Verify that pipeline state machine is strictly IDLE after queries."""
    pipeline = CraxyPipeline()
    assert pipeline.state == PipelineState.IDLE

    # Run math query
    out1 = pipeline.run_query("Calculate 25 * 4")
    assert pipeline.state == PipelineState.IDLE
    assert out1.state == PipelineState.IDLE

    # Run logic query
    out2 = pipeline.run_query("Execute algorithmic sorting: numbers = [4, 1, 3]")
    assert pipeline.state == PipelineState.IDLE
    assert out2.state == PipelineState.IDLE


def test_audit_conversation_history_rolling_window_ceiling():
    """Verify that multi-turn history retains only the last 3 turns in assembled context."""
    pipeline = CraxyPipeline()

    # Inject 50 simulated turns
    for i in range(50):
        pipeline.conversation_history.append((f"Query {i}", f"Response {i}"))

    assembled_prompt, estimated_tokens = pipeline._assemble_context("Final Query", [])

    # Context should only contain Query 47, 48, 49
    assert "Query 49" in assembled_prompt
    assert "Query 48" in assembled_prompt
    assert "Query 47" in assembled_prompt
    assert "Query 0" not in assembled_prompt
    assert "Query 40" not in assembled_prompt
    assert estimated_tokens < 200


def test_audit_benchmark_verifier_adversarial_rejection():
    """Verify that benchmark verifiers reject adversarial false-positive responses."""
    # 1. GSM8K: wrong numerical value
    assert not GSM8KBenchmark.verify_answer("The answer is 549.", "548")
    assert not GSM8KBenchmark.verify_answer("Calculation failed completely.", "548")

    # 2. MMLU: wrong factual answer
    assert not MMLUBenchmark.verify_answer("Quadratic memory is caused by dense MoE.", "FlashAttention")
    assert not MMLUBenchmark.verify_answer("", "FlashAttention")

    # 3. HumanEval: wrong algorithmic result
    assert not HumanEvalBenchmark.verify_answer("The sorted array is [1, 2, 8, 5]", "[1, 2, 5, 8]")
    assert not HumanEvalBenchmark.verify_answer("Syntax Error occurred", "[1, 2, 5, 8]")


def test_audit_sustained_multitool_allocator_stability():
    """Verify memory remains bounded (< 15.0 MB drift) over 150 mixed queries."""
    pipeline = CraxyPipeline()
    queries = [
        "Calculate 100 * 5 + 25",
        "Execute algorithmic sorting: numbers = [8, 3, 1, 9]",
        "What is the system design principle?",
        "Calculate 40 / 4 + 10",
    ]

    gc.collect()
    rss_initial = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

    for i in range(150):
        q = queries[i % len(queries)]
        pipeline.run_query(q)

    gc.collect()
    rss_final = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    drift = rss_final - rss_initial

    assert drift < 15.0
    assert rss_final < SYSTEM_LIMITS.max_ram_rss_mb
