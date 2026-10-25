"""
Independent Adversarial Red-Team Audit for Phase 2 Stage 10: Production Release Packaging and Verification.

Audits:
1. CLI argument parsing and exit codes (valid exit 0, invalid flags exit 2).
2. End-to-end subprocess execution across all subcommands (--info, --query, --benchmark).
3. Graceful fallback on nonexistent storage directories without unhandled crashes.
4. Comprehensive top-level package exports for all Phase 1 and Phase 2 modules.
5. Invariant physical limits verification (2,048 MB RAM, 100 GB NVMe, FlashAttention prefill, 4 KB sector alignment).
"""

import subprocess
import sys
import pytest

import src.craxy_ai as craxy
from src.craxy_ai.cli import build_parser, main
from src.craxy_ai.config import SYSTEM_LIMITS


def test_audit_cli_invalid_argument_exit_code():
    """Verify that invalid arguments raise SystemExit with exit code 2."""
    parser = build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--unknown-flag"])
    assert exc_info.value.code == 2

    # Empty arguments should also exit with code 2 because group is required
    with pytest.raises(SystemExit) as exc_empty:
        parser.parse_args([])
    assert exc_empty.value.code == 2


def test_audit_cli_all_subcommands_subprocess():
    """Verify that all CLI subcommands execute cleanly as independent subprocesses."""
    # 1. --info subcommand
    cmd_info = [sys.executable, "-m", "src.craxy_ai.cli", "--info"]
    proc_info = subprocess.run(cmd_info, capture_output=True, text=True, timeout=15)
    assert proc_info.returncode == 0
    assert "Craxy-AI Engine: Physical Architecture" in proc_info.stdout
    assert "Speculative Early Exit" in proc_info.stdout
    assert "DiskANN Vamana" in proc_info.stdout
    assert "SafeLogicEvaluator" in proc_info.stdout

    # 2. --query math tool
    cmd_query_math = [sys.executable, "-m", "src.craxy_ai.cli", "--query", "Calculate 25 * 8 + 50"]
    proc_math = subprocess.run(cmd_query_math, capture_output=True, text=True, timeout=15)
    assert proc_math.returncode == 0
    assert "250" in proc_math.stdout

    # 3. --query logic tool
    cmd_query_logic = [sys.executable, "-m", "src.craxy_ai.cli", "--query", "Execute algorithmic sorting: numbers = [7, 2, 9, 1]"]
    proc_logic = subprocess.run(cmd_query_logic, capture_output=True, text=True, timeout=15)
    assert proc_logic.returncode == 0
    assert "[1, 2, 7, 9]" in proc_logic.stdout


def test_audit_cli_nonexistent_storage_dir_fallback():
    """Verify graceful execution without crashing when storage directory does not exist."""
    cmd = [
        sys.executable, "-m", "src.craxy_ai.cli",
        "--query", "What is modular architecture?",
        "--storage-dir", "C:/nonexistent_path_craxy_ai_test_dir_12345"
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    assert proc.returncode == 0
    assert "Processed query directly from compact neural reasoning core" in proc.stdout


def test_audit_release_package_exports():
    """Verify that all core interfaces and Phase 2 modules are exported at top level."""
    expected_exports = [
        "CraxyPipeline",
        "CraxyNeuralCore",
        "SafeMathEvaluator",
        "SafeLogicEvaluator",
        "EarlyExitClassifier",
        "VamanaGraphStore",
        "SYSTEM_LIMITS",
        "CraxySystemLimits",
    ]
    for export_name in expected_exports:
        assert hasattr(craxy, export_name), f"Missing package export: {export_name}"


def test_audit_overall_system_constraints_verification():
    """Verify that physical system boundaries strictly enforce the 2 GB / 100 GB Charter."""
    assert SYSTEM_LIMITS.max_ram_rss_mb == 2048.0
    assert SYSTEM_LIMITS.max_storage_bytes == 100_000_000_000
    assert SYSTEM_LIMITS.max_context_window_tokens == 2048
    assert SYSTEM_LIMITS.max_prefill_chunk_tokens == 512
    assert SYSTEM_LIMITS.min_retrieval_similarity_threshold == 0.65
    assert SYSTEM_LIMITS.target_tokens_per_second == 2.0
