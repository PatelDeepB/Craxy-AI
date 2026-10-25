"""
Independent Adversarial Red-Team Audit for Phase 2 Stage 8 Safe AST Logic Evaluator.

Audits:
1. Rejection of memory exhaustion sequence multiplication bombs ([1] * 2_000_000).
2. Protection against sandbox evasion via dunder/subclass/MRO traversal.
3. Strict rejection of disallowed AST statement constructs (try/with/yield/global).
4. Deterministic step bounding and latency termination under nested loops.
5. Allocator drift (< 10.0 MB) over 1,500 complex algorithmic script evaluations.
"""

import gc
import os
import psutil
import pytest

from src.craxy_ai.tools.logic_evaluator import SafeLogicEvaluator
from src.craxy_ai.tools.logic_types import LogicEvaluationResult


def test_audit_adversarial_memory_exhaustion_multiplication_bomb():
    """Verify that oversized sequence multiplication bombs are blocked."""
    evaluator = SafeLogicEvaluator()

    # 1. List multiplication bomb
    res_list = evaluator.evaluate("arr = [1, 2] * 1_000_000")
    assert not res_list.is_success
    assert res_list.error_type == "MemoryError"

    # 2. String multiplication bomb
    res_str = evaluator.evaluate("s = 'xyz' * 1_000_000")
    assert not res_str.is_success
    assert res_str.error_type == "MemoryError"

    # 3. AugAssign multiplication bomb
    res_aug = evaluator.evaluate("s = 'ab'; s *= 2_000_000")
    assert not res_aug.is_success
    assert res_aug.error_type == "MemoryError"


def test_audit_sandbox_evasion_dunder_and_subclass_traversal():
    """Verify that sandbox escape attempts via object inspection are blocked."""
    evaluator = SafeLogicEvaluator()

    # 1. Class attribute access
    res_class = evaluator.evaluate("x = [].__class__")
    assert not res_class.is_success
    assert res_class.error_type == "SecurityViolationError"

    # 2. Subclasses traversal
    res_sub = evaluator.evaluate("x = ().__class__.__bases__[0].__subclasses__()")
    assert not res_sub.is_success
    assert res_sub.error_type == "SecurityViolationError"

    # 3. Globals access
    res_glob = evaluator.evaluate("g = globals()")
    assert not res_glob.is_success
    assert res_glob.error_type == "SecurityViolationError"

    # 4. Builtins access
    res_b = evaluator.evaluate("b = __builtins__")
    assert not res_b.is_success
    assert res_b.error_type == "SecurityViolationError"


def test_audit_disallowed_syntax_constructs():
    """Verify that dangerous Python syntax constructs are rejected at AST parse."""
    evaluator = SafeLogicEvaluator()

    # Try/except block
    res_try = evaluator.evaluate("try:\n    x = 1\nexcept:\n    x = 2")
    assert not res_try.is_success
    assert res_try.error_type == "SecurityViolationError"

    # With statement
    res_with = evaluator.evaluate("with 1:\n    pass")
    assert not res_with.is_success
    assert res_with.error_type == "SecurityViolationError"

    # Yield expression
    res_yield = evaluator.evaluate("def gen():\n    yield 1")
    assert not res_yield.is_success
    assert res_yield.error_type == "SecurityViolationError"

    # Global statement
    res_global = evaluator.evaluate("global x\nx = 1")
    assert not res_global.is_success
    assert res_global.error_type == "SecurityViolationError"


def test_audit_step_limit_tight_bounds():
    """Verify that nested loops terminate strictly at step limits without hanging."""
    evaluator = SafeLogicEvaluator(max_steps=250)
    code = """
total = 0
for i in range(100):
    for j in range(100):
        total += 1
"""
    result = evaluator.evaluate(code)
    assert not result.is_success
    assert result.error_type == "StepLimitExceededError"
    assert result.execution_time_us < 20000.0  # sub-20ms termination


def test_audit_sustained_stress_and_allocator_stability():
    """Verify memory remains constant (< 10.0 MB drift) over 1,500 complex algorithmic runs."""
    evaluator = SafeLogicEvaluator()
    code = """
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

res = quicksort([9, 3, 7, 1, 5, 2, 8, 4, 6])
"""
    gc.collect()
    rss_initial = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

    for _ in range(1500):
        res = evaluator.evaluate(code)
        assert res.is_success
        assert res.return_value == [1, 2, 3, 4, 5, 6, 7, 8, 9]

    gc.collect()
    rss_final = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    drift = rss_final - rss_initial

    assert drift < 10.0
    assert rss_final < 2048.0
