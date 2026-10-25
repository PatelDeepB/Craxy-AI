"""
Unit and release verification tests for Phase 2 Stage 10: Production Release Packaging.
"""

import subprocess
import sys
import pytest

from src.craxy_ai.cli import display_info, execute_benchmark, execute_query, main


def test_cli_info_execution(capsys):
    """Verify that CLI --info executes and prints Phase 2 telemetry."""
    ret = display_info()
    assert ret == 0

    captured = capsys.readouterr()
    assert "Craxy-AI Engine: Physical Architecture" in captured.out
    assert "2048.0 MB (2.0 GB)" in captured.out
    assert "Speculative Early Exit" in captured.out
    assert "DiskANN Vamana" in captured.out
    assert "SafeLogicEvaluator" in captured.out


def test_cli_query_with_logic_tool(capsys):
    """Verify that CLI executes algorithmic logic queries cleanly."""
    ret = execute_query("Execute algorithmic sorting: numbers = [8, 3, 5]")
    assert ret == 0

    captured = capsys.readouterr()
    assert "Craxy-AI Inference Output:" in captured.out
    assert "[1 executed]" in captured.out or "1 executed" in captured.out
    assert "[3, 5, 8]" in captured.out


def test_cli_benchmark_execution(capsys):
    """Verify that CLI executes standard and algorithmic benchmarks."""
    ret = execute_benchmark()
    assert ret == 0

    captured = capsys.readouterr()
    assert "GSM8K" in captured.out
    assert "HumanEval_Algorithmic" in captured.out
    assert "PASS (<= 2GB)" in captured.out


def test_cli_main_dispatch():
    """Verify that main() dispatches to correct subroutines."""
    assert main(["--info"]) == 0
    assert main(["--query", "Calculate 10 * 10"]) == 0
    assert main(["--benchmark"]) == 0


def test_cli_subprocess_execution():
    """Verify end-to-end execution of the CLI as a separate process."""
    cmd = [sys.executable, "-m", "src.craxy_ai.cli", "--query", "Execute algorithmic sorting: numbers = [4, 1, 2]"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)

    assert proc.returncode == 0
    assert "[1, 2, 4]" in proc.stdout
