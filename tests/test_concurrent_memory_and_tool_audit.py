"""
Unit tests for EXP-004 concurrent memory budget, DiskANN scaling, and sandboxed math evaluator.

Follows AAA pattern and tests critical safety and memory boundaries for the 2,048 MB constraint.
"""

import pytest
from experiments.exp004_concurrent_memory_and_tool_audit import (
    compute_vector_index_scaling_projections,
    execute_sandboxed_math,
    measure_concurrent_system_memory,
)


def test_concurrent_system_memory_fits_under_2gb() -> None:
    """Should verify that all components simultaneously resident in RAM fit under 2,048 MB."""
    # Arrange & Act
    budget = measure_concurrent_system_memory(
        backbone_params_b=1.5,
        embedding_params_m=33.0,
        context_tokens=2048
    )
    
    # Assert
    assert budget.fits_under_ceiling is True
    assert budget.available_headroom_mb > 500.0
    assert budget.measured_peak_rss_mb < 2048.0


def test_safe_math_evaluator_valid_arithmetic() -> None:
    """Should accurately compute complex arithmetic in microseconds without exec/eval."""
    # Arrange
    expr = "((250 * 12) + (150 * 8)) / 2"
    
    # Act
    result = execute_sandboxed_math(expr)
    
    # Assert
    assert result["success"] is True
    assert result["result"] == 2100.0
    assert result["error"] is None
    assert result["latency_us"] < 2000.0  # Under 2 ms


def test_safe_math_evaluator_blocks_code_injection() -> None:
    """Should securely reject malicious code injection attempts and dangerous functions."""
    # Arrange: Common code injection attacks
    malicious_inputs = [
        "__import__('os').system('dir')",
        "eval('2 + 2')",
        "open('secret.txt', 'w')",
        "import sys"
    ]
    
    for attack in malicious_inputs:
        # Act
        res = execute_sandboxed_math(attack)
        
        # Assert: Execution must be blocked
        assert res["success"] is False
        assert res["result"] is None
        assert "Disallowed" in res["error"] or "SyntaxError" in res["error"]


def test_safe_math_evaluator_handles_errors_gracefully() -> None:
    """Should gracefully return structured errors on divide-by-zero or overflow without crashing."""
    # Arrange & Act: Division by zero
    div_zero = execute_sandboxed_math("100 / 0")
    # Arrange & Act: Exponent overflow
    overflow = execute_sandboxed_math("2 ** 100000")
    
    # Assert
    assert div_zero["success"] is False
    assert "ZeroDivisionError" in div_zero["error"]
    
    assert overflow["success"] is False
    assert "OverflowError" in overflow["error"] or "ValueError" in overflow["error"]


def test_vector_index_scaling_proves_flat_scan_unviable_at_scale() -> None:
    """Should prove that flat scan I/O fails at 100 GB scale while DiskANN graph index succeeds."""
    # Arrange & Act
    projections = compute_vector_index_scaling_projections(embedding_dim=256)
    
    # Assert: Scale 0 (10k chunks) is viable for both
    assert projections[0]["flat_scan_viable"] is True
    assert projections[0]["diskann_viable"] is True
    
    # Scale 2 (50M chunks = ~100 GB scale): flat scan fails (> 50 ms), DiskANN passes (<= 50 ms)
    full_scale = projections[2]
    assert full_scale["flat_scan_viable"] is False
    assert full_scale["flat_scan_latency_ms"] > 5000.0  # > 5 seconds per query
    assert full_scale["diskann_viable"] is True
    assert full_scale["diskann_latency_ms"] < 50.0  # < 50 ms target
