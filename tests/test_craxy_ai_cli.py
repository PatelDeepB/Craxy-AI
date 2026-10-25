"""
Unit tests for the Craxy-AI Command-Line Interface (Milestone 6).

Validates:
- Invocation of --info flag displays physical boundaries and memory headroom.
- Invocation of --query flag executes end-to-end inference and tool dispatch.
- Invocation of --benchmark flag runs GSM8K, MMLU, and HumanEval suites.
- Strict exit code semantics (0 on success, non-zero on invalid arguments).
"""

import sys
import pytest
from unittest.mock import patch
import io

from src.craxy_ai.cli import main, build_parser


def test_cli_parser_build():
    """Verify that CLI parser registers required mutual exclusion groups."""
    parser = build_parser()
    assert parser.prog == "craxy-ai"

    # Must raise error if neither --info, --query, nor --benchmark is provided
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_cli_info_flag(capsys):
    """Verify that --info flag outputs memory limits and returns exit code 0."""
    exit_code = main(["--info"])
    assert exit_code == 0

    captured = capsys.readouterr().out
    assert "Craxy-AI Engine: Physical Architecture & Memory Boundaries" in captured
    assert "2048.0 MB" in captured
    assert "100.0 GB NVMe" in captured
    assert "512 tokens" in captured
    assert "Remaining Headroom" in captured


def test_cli_query_direct_reasoning(capsys):
    """Verify that --query flag processes prompts and returns exit code 0."""
    exit_code = main(["--query", "What is parameter decoupling in AI?"])
    assert exit_code == 0

    captured = capsys.readouterr().out
    assert "Craxy-AI Inference Output:" in captured
    assert "Peak Process RSS:" in captured
    assert "Tokens Generated:" in captured


def test_cli_query_with_math_tool(capsys):
    """Verify that --query flag executes mathematical expressions with precision."""
    exit_code = main(["--query", "Calculate 25 * 4 + 10 for inventory."])
    assert exit_code == 0

    captured = capsys.readouterr().out
    assert "Craxy-AI Inference Output:" in captured
    assert "110" in captured
    assert "[calc]" in captured


def test_cli_benchmark_execution(capsys):
    """Verify that --benchmark flag executes all standard evaluation suites."""
    exit_code = main(["--benchmark"])
    assert exit_code == 0

    captured = capsys.readouterr().out
    assert "Executing Craxy-AI Standardized Benchmarks" in captured
    assert "GSM8K" in captured
    assert "MMLU" in captured
    assert "HumanEval" in captured
    assert "PASS (<= 2GB)" in captured
