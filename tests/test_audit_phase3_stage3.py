"""Independent adversarial red-team audit of Phase 3 Stage 3: CraxyGGUFEngine.

Conducted by ULTRON 2 to red-team corrupt binary headers, context window
overflows, zero-token guardrails, multi-instance isolation, and memory drift.
"""

import os
import tempfile
import psutil
import pytest

from src.craxy_ai.core.gguf_engine import CraxyGGUFEngine
from src.craxy_ai.core.gguf_types import (
    ContextWindowOverflowError,
    GGUFEngineConfig,
    GGUFGenerationResult,
    InferenceExecutionError,
    MemoryBudgetExceededError,
)


@pytest.fixture
def temp_gguf_workspace():
    """Create a temporary sandbox containing corrupt and dummy model artifacts."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        corrupt_file = os.path.join(tmp_dir, "corrupt_model.gguf")
        with open(corrupt_file, "wb") as handle:
            handle.write(b"NOT_A_VALID_GGUF_HEADER_TRUNCATED_BINARY_DATA_XYZ_123456789")

        yield tmp_dir, corrupt_file


def test_audit_corrupt_gguf_binary_header_rejection(temp_gguf_workspace):
    """Adversarial check: ensure invalid binary headers raise typed error without crash."""
    _, corrupt_path = temp_gguf_workspace
    config = GGUFEngineConfig(model_path=corrupt_path)
    engine = CraxyGGUFEngine(config=config)

    with pytest.raises(InferenceExecutionError):
        engine.load_model()

    assert not engine.is_loaded


def test_audit_context_window_overflow_trapping():
    """Adversarial check: ensure prompt tokens exceeding context ceiling are rejected."""
    # Context window set to a tight 32 tokens
    config = GGUFEngineConfig(context_window_tokens=32)
    engine = CraxyGGUFEngine(config=config)
    engine.load_model()

    # Prompt with ~50 words which exceeds 32 tokens
    long_prompt = "word " * 50

    with pytest.raises(ContextWindowOverflowError):
        engine.generate(long_prompt, max_tokens=10)


def test_audit_zero_max_tokens_guard():
    """Adversarial check: ensure requesting zero tokens avoids division by zero in TPS."""
    engine = CraxyGGUFEngine()
    engine.load_model()

    result: GGUFGenerationResult = engine.generate("Hello Craxy-AI", max_tokens=0)
    assert result.completion_tokens == 0
    assert result.tokens_per_second >= 0.0
    assert result.finish_reason in ("length", "stop")


def test_audit_multi_instance_state_isolation():
    """Adversarial check: ensure multiple engine instances maintain isolated state."""
    config_a = GGUFEngineConfig(context_window_tokens=1024, max_ram_budget_mb=1600.0)
    config_b = GGUFEngineConfig(context_window_tokens=4096, max_ram_budget_mb=1800.0)

    engine_a = CraxyGGUFEngine(config=config_a)
    engine_b = CraxyGGUFEngine(config=config_b)

    engine_a.load_model()
    engine_b.load_model()

    stat_a = engine_a.get_memory_status()
    stat_b = engine_b.get_memory_status()

    assert stat_a["budget_mb"] == 1600.0
    assert stat_b["budget_mb"] == 1800.0

    engine_a.unload_model()
    assert not engine_a.is_loaded
    assert engine_b.is_loaded

    engine_b.unload_model()
    assert not engine_b.is_loaded


def test_audit_sustained_100_generation_cycles_memory_drift():
    """Adversarial check: ensure 100 sustained generation cycles exhibit < 5.0 MB drift."""
    engine = CraxyGGUFEngine()
    engine.load_model()
    proc = psutil.Process()
    rss_start = proc.memory_info().rss / (1024 * 1024)

    for i in range(100):
        engine.generate(f"Query prompt iteration cycle {i}", max_tokens=8)

    rss_end = proc.memory_info().rss / (1024 * 1024)
    rss_drift = abs(rss_end - rss_start)

    # Must maintain strict memory ceiling under 5.0 MB drift
    assert rss_drift < 5.0
    engine.unload_model()
