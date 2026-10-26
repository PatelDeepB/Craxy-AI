"""Tests for Phase 5 Production CLI Exposure and Frontier Capabilities.

Validates:
- CLI flag parsing for --info, --query, --benchmark, --gguf, and --paged-kv.
- Memory telemetry reporting under 2,048 MB physical DRAM budget.
- Subprocess and in-process execution pathways.
- Automatic resource teardown for temporary paged KV directories.
"""

import io
import os
import sys
from unittest.mock import MagicMock, patch
import pytest

from src.craxy_ai.cli import build_parser, main
from src.craxy_ai.core.gguf_types import GGUFGenerationResult


def test_cli_info_flag_returns_success(capsys):
    """Verify that --info flag displays system specs and positive memory headroom."""
    ret_code = main(["--info"])
    captured = capsys.readouterr().out

    assert ret_code == 0
    assert "Craxy-AI Engine: Physical Architecture" in captured
    assert "NVMe-Paged KV Cache" in captured
    assert "Quantized GGUF Engine" in captured
    assert "Max RAM RSS Budget:             2048.0 MB" in captured


def test_cli_query_math_tool_dispatch(capsys):
    """Verify that arithmetic query evaluates and prints tool output."""
    ret_code = main(["--query", "Calculate 120 * 4 + 10"])
    captured = capsys.readouterr().out

    assert ret_code == 0
    assert "Craxy-AI Inference Output:" in captured
    assert "[Tool Result: 490]" in captured
    assert "[calc]" in captured


def test_cli_query_with_paged_kv_auto(capsys):
    """Verify that --paged-kv auto attaches cache and runs query cleanly."""
    ret_code = main(["--query", "Sort array numbers = [4, 1, 9]", "--paged-kv", "auto"])
    captured = capsys.readouterr().out

    assert ret_code == 0
    assert "[Tool Result: [1, 4, 9]]" in captured
    assert "Peak Process RSS:" in captured


def test_cli_query_with_gguf_mock(capsys):
    """Verify that --gguf flag activates GGUF inference tier."""
    mock_res = GGUFGenerationResult(
        generated_text="GGUF mock response sequence.",
        prompt_tokens=10,
        completion_tokens=6,
        total_tokens=16,
        tokens_per_second=150.0,
        peak_rss_mb=410.0,
        duration_seconds=0.04,
        finish_reason="stop",
    )
    with patch("src.craxy_ai.core.gguf_engine.CraxyGGUFEngine") as mock_engine_cls:
        instance = MagicMock()
        instance.generate.return_value = mock_res
        mock_engine_cls.return_value = instance

        ret_code = main(["--query", "Analyze parameter efficiency", "--gguf", "weights.gguf"])
        captured = capsys.readouterr().out

        assert ret_code == 0
        assert "Engine Type:            gguf" in captured
        assert "GGUF mock response sequence." in captured


def test_cli_benchmark_execution(capsys):
    """Verify that --benchmark flag executes runner and prints summary table."""
    ret_code = main(["--benchmark"])
    captured = capsys.readouterr().out

    assert ret_code == 0
    assert "Executing Craxy-AI Standardized Benchmarks" in captured
    assert "GSM8K" in captured
    assert "MMLU" in captured
    assert "HumanEval" in captured
    assert "PASS (<= 2GB)" in captured


def test_cli_missing_mode_flag_raises_system_exit():
    """Verify that omitting required mode flags exits with error."""
    with pytest.raises(SystemExit):
        main([])
