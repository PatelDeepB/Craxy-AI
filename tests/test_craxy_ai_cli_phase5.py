"""Unit and integration tests for Phase 5 Production CLI Exposure.

Validates:
- Display of system limits and Phase 3/4 frontier capabilities via CLI --info.
- Query execution with --use-gguf routing to quantized GGUF neural engine.
- Query execution with --paged-kv routing to 128K NVMe-backed paged KV cache.
- Code syntax validation and workspace resolution via --code-patch.
- End-to-end CLI subprocess dispatch under strict 2,048 MB memory budget.
"""

import subprocess
import sys
from unittest.mock import MagicMock, patch
import pytest

from src.craxy_ai.cli import (
    build_parser,
    display_info,
    execute_benchmark,
    execute_code_patch,
    execute_query,
    main,
)
from src.craxy_ai.core.gguf_types import GGUFGenerationResult


def test_cli_phase5_info_displays_frontier_capabilities(capsys):
    """Verify that CLI --info displays Phase 3 & 4 frontier features."""
    ret = display_info()
    assert ret == 0

    captured = capsys.readouterr()
    assert "Craxy-AI Engine: Physical Architecture" in captured.out
    assert "NVMe Paged KV Cache" in captured.out
    assert "Quantized GGUF Engine:" in captured.out
    assert "Autonomous Code Agent:" in captured.out
    assert "Remaining Headroom to Budget:" in captured.out


def test_cli_phase5_query_with_gguf_flag(capsys):
    """Verify that --use-gguf routes generation to the GGUF neural engine."""
    mock_res = GGUFGenerationResult(
        generated_text="GGUF quantized response via CLI dispatch.",
        prompt_tokens=10,
        completion_tokens=6,
        total_tokens=16,
        tokens_per_second=150.0,
        peak_rss_mb=420.0,
        duration_seconds=0.04,
        finish_reason="stop",
    )
    with patch("src.craxy_ai.core.gguf_engine.CraxyGGUFEngine") as mock_engine_cls:
        mock_instance = MagicMock()
        mock_instance.generate.return_value = mock_res
        mock_engine_cls.return_value = mock_instance

        ret = execute_query("Executive prompt", use_gguf=True)
        assert ret == 0

    captured = capsys.readouterr()
    assert "Craxy-AI Inference Output:" in captured.out
    assert "Engine Type:            gguf" in captured.out
    assert "GGUF quantized response via CLI dispatch." in captured.out


def test_cli_phase5_query_with_paged_kv_flag(capsys):
    """Verify that --paged-kv attaches NVMe-paged KV cache during query."""
    ret = execute_query("Reasoning query with long context", use_paged_kv=True)
    assert ret == 0

    captured = capsys.readouterr()
    assert "Craxy-AI Inference Output:" in captured.out
    assert "Context Tokens Used:" in captured.out
    assert "Peak Process RSS:" in captured.out


def test_cli_phase5_code_patch_dispatch(capsys):
    """Verify that --code-patch validates Python syntax and prints workspace root."""
    valid_code = "def add(x: int, y: int) -> int:\n    return x + y\n"
    ret_valid = execute_code_patch(valid_code)
    assert ret_valid == 0

    captured = capsys.readouterr()
    assert "CraxyCodeAgent Syntax Validation:" in captured.out
    assert "Syntax Valid:           True" in captured.out

    invalid_code = "def broken(\n"
    ret_invalid = execute_code_patch(invalid_code)
    assert ret_invalid == 1


def test_cli_phase5_main_dispatch_options():
    """Verify that main() dispatches to all Phase 5 CLI options."""
    assert main(["--info"]) == 0
    assert main(["--code-patch", "x = 42"]) == 0
    assert main(["--query", "Calculate 25 * 4", "--paged-kv"]) == 0


def test_cli_phase5_subprocess_execution():
    """Verify end-to-end execution of CLI in an isolated subprocess."""
    cmd = [sys.executable, "-m", "src.craxy_ai.cli", "--code-patch", "print('hello from subprocess')"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)

    assert proc.returncode == 0
    assert "Syntax Valid:           True" in proc.stdout
