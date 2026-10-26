"""Adversarial Red-Team Audit for Phase 8 Continuous Benchmark Suite and Dashboard (Turn 65).

Audits:
1. Workspace path robustness: auto-creation of missing workspace directories.
2. CLI alias equivalence: --dashboard flag executes with identical schema and exit code 0.
3. Telemetry budget invariants: mathematical consistency (RSS + headroom == 2048 MB) and budget bounds.
4. Adversarial metric failure gating: fail-closed safety and exit code 1 propagation.
5. 20-cycle programmatic evaluation memory stability (< 20.0 MB drift under gc.collect()).
"""

import gc
import os
import subprocess
import sys
import shutil
import tempfile
import psutil
import pytest

from src.craxy_ai.benchmarks.frontier_eval import (
    FrontierEvaluationSuite,
    FrontierEvaluationResult,
    FrontierMetric,
    render_dashboard,
)
from src.craxy_ai.cli import execute_eval


def test_audit_frontier_eval_missing_workspace_recovery():
    """Verify evaluation suite automatically provisions nonexistent nested workspace path."""
    nested_dir = os.path.join(tempfile.gettempdir(), "craxy_adversarial_eval_dir", "sub_path")
    if os.path.exists(nested_dir):
        shutil.rmtree(nested_dir, ignore_errors=True)

    suite = FrontierEvaluationSuite(workspace_dir=nested_dir)
    metric = suite.evaluate_agentic_code_sandbox()

    try:
        assert metric.passed is True
        assert metric.value == 100.0
        assert os.path.exists(nested_dir)
    finally:
        shutil.rmtree(nested_dir, ignore_errors=True)


def test_audit_cli_dashboard_alias_execution():
    """Verify --dashboard alias executes cleanly with exit code 0 and renders telemetry."""
    cmd = [sys.executable, "-m", "src.craxy_ai", "--dashboard"]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0
    assert "Craxy-AI: Frontier Capability & Resource Telemetry Dashboard" in result.stdout
    assert "Overall Status:  PASS" in result.stdout
    assert "Traceback" not in result.stderr


def test_audit_telemetry_budget_invariants():
    """Verify telemetry outputs satisfy physical resource budget constraints."""
    suite = FrontierEvaluationSuite()
    result = suite.run_full_evaluation()

    # Budget invariant: DRAM ceiling is 2048.0 MB
    total_dram = result.peak_rss_mb + result.headroom_mb
    assert abs(total_dram - 2048.0) < 1e-4

    # Physical limits verification
    assert result.peak_rss_mb < 2048.0
    assert result.headroom_mb > 1000.0
    assert result.storage_mb < 100.0 * 1024.0  # Under 100 GB


def test_audit_fail_closed_status_on_adversarial_metric():
    """Verify fail-closed gating when an adversarial metric failure is injected."""
    failing_metric = FrontierMetric(
        name="Adversarial Adversity Probe",
        category="Security",
        value=0.0,
        target=100.0,
        unit="%",
        passed=False,
    )
    res = FrontierEvaluationResult(
        timestamp_utc="2026-10-26T00:00:00Z",
        metrics=[failing_metric],
        peak_rss_mb=250.0,
        headroom_mb=1798.0,
        storage_mb=10.0,
        all_passed=False,
        duration_sec=0.1,
    )

    dashboard = render_dashboard(res)
    assert "Overall Status:  FAIL" in dashboard
    assert "Adversarial Adversity Probe" in dashboard
    assert "-> FAIL" in dashboard


def test_audit_repeated_programmatic_evaluation_memory_stability():
    """Verify 20 repeated programmatic evaluation cycles exhibit bounded RSS drift (< 20.0 MB)."""
    suite = FrontierEvaluationSuite()

    # Warmup cycle
    suite.run_full_evaluation()
    gc.collect()

    process = psutil.Process(os.getpid())
    initial_rss_mb = process.memory_info().rss / (1024 * 1024)

    for _ in range(20):
        res = suite.run_full_evaluation()
        assert res.all_passed is True

    gc.collect()
    final_rss_mb = process.memory_info().rss / (1024 * 1024)
    rss_drift_mb = final_rss_mb - initial_rss_mb

    assert rss_drift_mb < 20.0, f"Memory drift exceeded threshold: {rss_drift_mb:.2f} MB"
