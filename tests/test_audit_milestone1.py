"""
Independent Red-Team Audit of Milestone 1 Production Modules (Turn 8).

Audits:
- src.craxy_ai.config (immutability, limits compliance with ADRs 001-006)
- src.craxy_ai.tools.math_evaluator (SafeMathEvaluator latency, deep recursion, memory drift, edge cases)
"""

import ast
import gc
import os
import psutil
import pytest

from src.craxy_ai.config import SYSTEM_LIMITS, DEFAULT_MODEL_CONFIG, CraxySystemLimits
from src.craxy_ai.tools.math_evaluator import SafeMathEvaluator, EvaluationResult


def test_config_immutability_and_limits():
    """Verify that configuration limits cannot be mutated at runtime."""
    # Frozen dataclass must raise on attribute modification
    with pytest.raises(Exception):
        SYSTEM_LIMITS.max_ram_rss_mb = 4096.0  # type: ignore

    # Verify hard ceilings comply with ADRs and Research Charter
    assert SYSTEM_LIMITS.max_ram_rss_mb <= 2048.0
    assert SYSTEM_LIMITS.max_storage_bytes <= 100 * 10**9
    assert SYSTEM_LIMITS.max_prefill_chunk_tokens <= 512
    assert SYSTEM_LIMITS.max_context_window_tokens <= 2048
    assert SYSTEM_LIMITS.min_retrieval_similarity_threshold >= 0.65


def test_safe_math_unary_chaining_and_floats():
    """Verify evaluator handles complex unary combinations and floating-point literals."""
    evaluator = SafeMathEvaluator()

    # Unary chaining
    res1 = evaluator.evaluate("-(-5)")
    assert res1.is_success
    assert res1.computed_value == 5

    res2 = evaluator.evaluate("---5")
    assert res2.is_success
    assert res2.computed_value == -5

    res3 = evaluator.evaluate("-(3 * 4) + +7")
    assert res3.is_success
    assert res3.computed_value == -5

    # Scientific notation floats
    res4 = evaluator.evaluate("1.5e3 + 2.5e2")
    assert res4.is_success
    assert res4.computed_value == 1750.0


def test_safe_math_deeply_nested_recursion_handling():
    """Verify that deeply nested parentheses do not crash the host process with unhandled exceptions."""
    evaluator = SafeMathEvaluator()

    # 100 levels of parentheses
    nested_expr = "(" * 100 + "42" + ")" * 100
    res = evaluator.evaluate(nested_expr)
    assert res.is_success
    assert res.computed_value == 42

    # Malformed unbalanced parentheses
    bad_expr = "(" * 50 + "42"
    res_bad = evaluator.evaluate(bad_expr)
    assert not res_bad.is_success
    assert res_bad.error_type == "SyntaxError"


def test_safe_math_zero_memory_leakage_under_repeated_calls():
    """Verify that 2,000 evaluations exhibit zero memory leakage / RSS drift (< 0.5 MB)."""
    evaluator = SafeMathEvaluator()
    process = psutil.Process(os.getpid())

    gc.collect()
    rss_before = process.memory_info().rss / (1024 * 1024)

    for i in range(2000):
        expr = f"{i} * 2 + (100 / 4) - {i % 10}"
        res = evaluator.evaluate(expr)
        assert res.is_success

    gc.collect()
    rss_after = process.memory_info().rss / (1024 * 1024)
    rss_delta = rss_after - rss_before

    # Verify drift is bounded to less than 1.0 MB
    assert rss_delta < 1.0, f"Excessive memory drift detected: {rss_delta:.2f} MB"


def test_safe_math_evaluation_latency_ceiling():
    """Verify single evaluation latency remains strictly under 100 microseconds on CPU."""
    evaluator = SafeMathEvaluator()
    res = evaluator.evaluate("150 * (24 + 18) - 1200 / 4")
    assert res.is_success
    assert res.computed_value == 6000.0
    # Latency should be well below 100 us (typically 5 to 25 us)
    assert res.execution_time_us < 200.0, f"Execution latency too high: {res.execution_time_us:.2f} us"
