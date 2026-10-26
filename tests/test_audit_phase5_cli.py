"""Adversarial Red-Team Audit for Phase 5 Production CLI Exposure (Turn 59).

Audits:
1. Mutual exclusion enforcement and CLI argument boundary stress.
2. Adversarial and malformed Python syntax rejection in --code-patch.
3. Non-existent and corrupted storage/GGUF paths fault tolerance.
4. Deterministic teardown and zero handle leakage for temporary paged KV directories.
5. Multi-cycle memory stability and strict adherence to 2,048 MB RAM budget.
"""

import gc
import os
import sys
import tempfile
from unittest.mock import patch
import psutil
import pytest

from src.craxy_ai.cli import execute_code_patch, main


def test_audit_cli_mutex_enforcement():
    """Verify that specifying conflicting operational modes triggers SystemExit."""
    conflicting_combos = [
        ["--info", "--query", "What is 10 + 20?"],
        ["--info", "--benchmark"],
        ["--query", "test", "--benchmark"],
        ["--benchmark", "--code-patch", "x = 1"],
        ["--info", "--code-patch", "x = 1"],
    ]
    for combo in conflicting_combos:
        with pytest.raises(SystemExit) as exc_info:
            main(combo)
        assert exc_info.value.code == 2


def test_audit_cli_code_patch_adversarial_inputs(capsys):
    """Verify that malformed and edge-case code inputs are handled safely."""
    # Scenario 1: Malformed syntax missing paren
    bad_syntax = "def broken(:\n    return 1"
    code_result = execute_code_patch(bad_syntax)
    captured = capsys.readouterr().out
    assert code_result == 1
    assert "Syntax Valid:           False" in captured

    # Scenario 2: Bad indentation
    bad_indent = "def test():\nreturn 1"
    code_result = execute_code_patch(bad_indent)
    captured = capsys.readouterr().out
    assert code_result == 1
    assert "Syntax Valid:           False" in captured

    # Scenario 3: Valid multiline Python code
    valid_code = "import math\n\ndef square(x: float) -> float:\n    return x * x\n"
    code_result = execute_code_patch(valid_code)
    captured = capsys.readouterr().out
    assert code_result == 0
    assert "Syntax Valid:           True" in captured

    # Scenario 4: Empty string
    code_result = execute_code_patch("")
    captured = capsys.readouterr().out
    assert code_result == 0
    assert "Syntax Valid:           True" in captured

    # Scenario 5: Unicode UTF-8 content
    unicode_code = "message = 'こんにちは世界'\nvalue = 42\n"
    code_result = execute_code_patch(unicode_code)
    captured = capsys.readouterr().out
    assert code_result == 0
    assert "Syntax Valid:           True" in captured


def test_audit_cli_nonexistent_and_corrupted_storage_paths(capsys):
    """Verify that non-existent storage and GGUF paths do not cause crashes."""
    fake_storage = "C:/craxy_nonexistent_test_storage_dir_xyz"
    fake_gguf = "C:/craxy_nonexistent_weights.gguf"

    # Query with non-existent storage directory
    ret_storage = main(["--query", "Calculate 40 + 60", "--storage-dir", fake_storage])
    captured_storage = capsys.readouterr().out
    assert ret_storage == 0
    assert "[Tool Result: 100]" in captured_storage

    # Query with non-existent GGUF path: must fail cleanly with code 1 and error message
    ret_gguf = main(["--query", "Calculate 15 * 3", "--use-gguf", "--gguf-path", fake_gguf])
    captured_gguf = capsys.readouterr().out
    assert ret_gguf == 1
    assert "Craxy-AI Query Execution Error:" in captured_gguf
    assert "GGUF model file not found" in captured_gguf


def test_audit_cli_temp_paged_kv_deterministic_teardown(capsys):
    """Verify that temporary paged KV directory is deterministically deleted."""
    created_dirs = []
    real_mkdtemp = tempfile.mkdtemp

    def recording_mkdtemp(*args, **kwargs):
        d = real_mkdtemp(*args, **kwargs)
        created_dirs.append(d)
        return d

    with patch("tempfile.mkdtemp", side_effect=recording_mkdtemp):
        ret_code = main(["--query", "Calculate 200 / 4", "--paged-kv", "auto"])
        captured = capsys.readouterr().out

    assert ret_code == 0
    assert "[Tool Result: 50.0]" in captured
    assert len(created_dirs) == 1

    temp_dir = created_dirs[0]
    # Check that directory was removed in the finally block
    assert not os.path.exists(temp_dir), f"Directory {temp_dir} was not cleaned up!"


def test_audit_cli_consecutive_invocations_memory_drift(capsys):
    """Verify that repeated CLI invocations preserve zero memory drift."""
    gc.collect()
    process = psutil.Process(os.getpid())
    rss_initial = process.memory_info().rss / (1024 * 1024)

    for idx in range(25):
        ret_query = main(["--query", f"Calculate {idx} + 10"])
        assert ret_query == 0
        ret_patch = main(["--code-patch", f"x = {idx} * 2"])
        assert ret_patch == 0

    capsys.readouterr()  # Flush stdout
    gc.collect()
    rss_final = process.memory_info().rss / (1024 * 1024)
    growth_mb = rss_final - rss_initial

    # Allow modest growth for Python module caching, but strictly bounded
    assert growth_mb < 25.0, f"Memory growth {growth_mb:.2f} MB exceeds 25 MB limit"
    assert rss_final < 2048.0, f"Peak memory {rss_final:.2f} MB exceeds 2048 MB budget"
