"""Frontier Evaluation Suite and Telemetry Dashboard Verification Tests (Turn 64).

Validates:
1. Full evaluation execution across math reasoning, 128K context, and agentic coding.
2. Capability metric structure, target thresholds, and units.
3. Telemetry dashboard formatting and absence of disallowed em dash characters.
4. JSON serialization of evaluation results.
5. Subprocess CLI invocation of craxy-ai --eval.
6. Strict physical memory ceiling compliance (DRAM RSS <= 2,048 MB).
"""

import json
import os
import subprocess
import sys
import psutil
import pytest

from src.craxy_ai.benchmarks.frontier_eval import (
    FrontierEvaluationResult,
    FrontierEvaluationSuite,
    render_dashboard,
)
from src.craxy_ai.config import SYSTEM_LIMITS


def test_frontier_eval_suite_execution():
    """Verify suite.run_full_evaluation() executes successfully with all passed."""
    suite = FrontierEvaluationSuite()
    result = suite.run_full_evaluation()

    assert isinstance(result, FrontierEvaluationResult)
    assert result.all_passed is True
    assert result.peak_rss_mb < SYSTEM_LIMITS.max_ram_rss_mb
    assert result.headroom_mb > 500.0
    assert result.duration_sec > 0.0
    assert len(result.metrics) == 3


def test_frontier_eval_metrics_structure():
    """Verify all individual metrics contain proper categories, targets, and units."""
    suite = FrontierEvaluationSuite()
    result = suite.run_full_evaluation()

    categories = {m.category for m in result.metrics}
    assert "Reasoning" in categories
    assert "Context" in categories
    assert "Coding" in categories

    for metric in result.metrics:
        assert metric.value >= metric.target
        assert metric.unit == "%"
        assert metric.passed is True


def test_frontier_eval_dashboard_formatting():
    """Verify render_dashboard outputs structured headers and contains zero em dashes."""
    suite = FrontierEvaluationSuite()
    result = suite.run_full_evaluation()
    dashboard = render_dashboard(result)

    assert "Craxy-AI: Frontier Capability & Resource Telemetry Dashboard" in dashboard
    assert "Overall Status:  PASS" in dashboard
    assert "Physical Resource Telemetry (Budget <= 2048 MB):" in dashboard
    assert "Process Peak DRAM (RSS):" in dashboard
    assert "\u2014" not in dashboard, "Disallowed em dash found in rendered dashboard"


def test_frontier_eval_to_dict_serialization():
    """Verify to_dict returns valid JSON-serializable dictionary."""
    suite = FrontierEvaluationSuite()
    result = suite.run_full_evaluation()
    data = result.to_dict()

    assert isinstance(data, dict)
    assert data["all_passed"] is True
    assert "metrics" in data
    assert len(data["metrics"]) == 3

    # Ensure clean json serialization
    json_str = json.dumps(data)
    assert len(json_str) > 100


def test_frontier_eval_cli_subprocess():
    """Verify python -m src.craxy_ai --eval executes cleanly with exit code 0."""
    cmd = [sys.executable, "-m", "src.craxy_ai", "--eval"]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

    assert res.returncode == 0
    stdout = res.stdout
    assert "Craxy-AI: Frontier Capability & Resource Telemetry Dashboard" in stdout
    assert "Overall Status:  PASS" in stdout
    assert "GSM8K Math Reasoning" in stdout
    assert "128K NVMe Context Paging" in stdout


def test_frontier_eval_memory_headroom_guarantee():
    """Verify sustained execution keeps resident DRAM headroom strictly > 500 MB."""
    suite = FrontierEvaluationSuite()
    for _ in range(3):
        res = suite.run_full_evaluation()
        assert res.peak_rss_mb < 2048.0
        assert res.headroom_mb > 500.0
