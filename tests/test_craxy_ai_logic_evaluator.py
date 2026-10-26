"""
Unit and safety tests for SafeLogicEvaluator.
"""

import gc
import os
import psutil
import pytest

from src.craxy_ai.tools.logic_evaluator import SafeLogicEvaluator
from src.craxy_ai.tools.logic_types import (
    LogicEvaluationResult,
    SecurityViolationError,
    StepLimitExceededError,
)


def test_basic_algorithmic_execution():
    """Verify variables, loops, conditionals, and return values."""
    evaluator = SafeLogicEvaluator()
    code = """
def sum_evens(nums):
    total = 0
    for n in nums:
        if n % 2 == 0:
            total += n
    return total

res = sum_evens([1, 2, 3, 4, 5, 6, 7, 8])
"""
    result = evaluator.evaluate(code)
    assert result.is_success
    assert result.error_type is None
    assert result.steps_executed > 10


def test_fibonacci_and_entry_point():
    """Verify function definition and direct entry point invocation."""
    evaluator = SafeLogicEvaluator()
    code = """
def fib(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b
"""
    result = evaluator.evaluate(code, entry_point="fib", args=[10])
    assert result.is_success
    assert result.return_value == 55
    assert result.execution_time_us < 10000.0


def test_list_comprehension_and_builtins():
    """Verify list comprehensions, sorting, and allowed builtins."""
    evaluator = SafeLogicEvaluator()
    code = """
words = ["banana", "apple", "cherry", "date"]
upper_words = [w.upper() for w in words]
sorted_words = sorted(upper_words)
"""
    result = evaluator.evaluate(code)
    assert result.is_success
    assert result.error_type is None


def test_security_rejection_of_imports_and_builtins():
    """Verify that imports, eval, and system calls are strictly blocked."""
    evaluator = SafeLogicEvaluator()

    # Import statement
    res_import = evaluator.evaluate("import os")
    assert not res_import.is_success
    assert res_import.error_type == "SecurityViolationError"

    # From import statement
    res_from = evaluator.evaluate("from sys import exit")
    assert not res_from.is_success
    assert res_from.error_type == "SecurityViolationError"

    # Dangerous builtin
    res_eval = evaluator.evaluate("eval('1 + 1')")
    assert not res_eval.is_success
    assert res_eval.error_type == "SecurityViolationError"

    # Open file
    res_open = evaluator.evaluate("open('test.txt', 'w')")
    assert not res_open.is_success
    assert res_open.error_type == "SecurityViolationError"

    # Dunder attribute access
    res_dunder = evaluator.evaluate("x = (1).__class__.__name__")
    assert not res_dunder.is_success
    assert res_dunder.error_type == "SecurityViolationError"


def test_step_limit_prevents_infinite_loop():
    """Verify that infinite loops are halted deterministically by step limits."""
    evaluator = SafeLogicEvaluator(max_steps=500)
    code = """
x = 0
while True:
    x += 1
"""
    result = evaluator.evaluate(code)
    assert not result.is_success
    assert result.error_type == "StepLimitExceededError"
    assert "Step limit of 500 exceeded" in result.error_message


def test_recursion_depth_limit():
    """Verify that infinite recursion is halted safely."""
    evaluator = SafeLogicEvaluator(max_call_depth=10)
    code = """
def inf_rec(x):
    return inf_rec(x + 1)

inf_rec(0)
"""
    result = evaluator.evaluate(code)
    assert not result.is_success
    assert result.error_type == "RecursionError"


def test_allocator_stability_over_many_runs():
    """Verify memory remains constant (< 5.0 MB drift) over 1,000 evaluations."""
    evaluator = SafeLogicEvaluator()
    code = """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

res = bubble_sort([5, 1, 4, 2, 8])
"""
    gc.collect()
    rss_initial = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

    for _ in range(1000):
        res = evaluator.evaluate(code)
        assert res.is_success

    gc.collect()
    rss_final = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    drift = rss_final - rss_initial

    assert drift < 5.0
    assert rss_final < 2048.0
