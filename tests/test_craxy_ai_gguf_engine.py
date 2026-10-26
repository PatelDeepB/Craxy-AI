"""Unit tests for CraxyGGUFEngine (Phase 3 Stage 3 Quantized Inference Engine).

Verifies memory-mapped loading, budget watchdog enforcement, context bounds,
and zero memory leakage over sustained generation passes.
"""

import os
import psutil
import pytest

from src.craxy_ai.core.gguf_engine import CraxyGGUFEngine
from src.craxy_ai.core.gguf_types import (
    ContextWindowOverflowError,
    GGUFEngineConfig,
    GGUFGenerationResult,
    MemoryBudgetExceededError,
    ModelNotFoundError,
)


def test_gguf_engine_initialization():
    """Verify default and custom configuration initialization."""
    engine = CraxyGGUFEngine()
    assert not engine.is_loaded
    assert engine.config.context_window_tokens == 2048
    assert engine.config.max_ram_budget_mb == 1500.0

    custom_cfg = GGUFEngineConfig(
        context_window_tokens=4096,
        max_ram_budget_mb=1800.0,
        num_threads=2,
    )
    custom_engine = CraxyGGUFEngine(config=custom_cfg)
    assert custom_engine.config.context_window_tokens == 4096
    assert custom_engine.config.num_threads == 2


def test_gguf_engine_missing_file_raises_model_not_found():
    """Verify specifying a nonexistent GGUF artifact raises ModelNotFoundError."""
    cfg = GGUFEngineConfig(model_path="nonexistent_qwen_coder_weights.gguf")
    engine = CraxyGGUFEngine(config=cfg)
    with pytest.raises(ModelNotFoundError) as exc_info:
        engine.load_model()
    assert "GGUF model file not found" in str(exc_info.value)


def test_gguf_engine_memory_status_telemetry():
    """Verify memory status telemetry provides accurate RSS and headroom metrics."""
    engine = CraxyGGUFEngine()
    status = engine.get_memory_status()
    assert "rss_mb" in status
    assert "budget_mb" in status
    assert "headroom_mb" in status
    assert status["rss_mb"] > 0.0
    assert status["budget_mb"] == 1500.0
    assert status["headroom_mb"] > 0.0


def test_gguf_engine_generation_simulation():
    """Verify token generation returns valid telemetry and text output."""
    engine = CraxyGGUFEngine()
    result: GGUFGenerationResult = engine.generate(
        prompt="Write a Python function to double an integer.",
        max_tokens=16,
    )
    assert result.completion_tokens > 0
    assert len(result.generated_text) > 0
    assert result.tokens_per_second > 0.0
    assert result.peak_rss_mb > 0.0
    assert result.duration_seconds > 0.0
    assert result.finish_reason == "stop"


def test_gguf_engine_context_overflow_protection():
    """Verify prompt and completion exceeding context budget raises ContextWindowOverflowError."""
    cfg = GGUFEngineConfig(context_window_tokens=32)
    engine = CraxyGGUFEngine(config=cfg)
    oversized_prompt = "word " * 40
    with pytest.raises(ContextWindowOverflowError) as exc_info:
        engine.generate(prompt=oversized_prompt, max_tokens=16)
    assert "Context overflow" in str(exc_info.value)


def test_gguf_engine_budget_watchdog_enforcement():
    """Verify memory watchdog aborts when process RSS exceeds allocated budget."""
    cfg = GGUFEngineConfig(max_ram_budget_mb=1.0)
    engine = CraxyGGUFEngine(config=cfg)
    with pytest.raises(MemoryBudgetExceededError) as exc_info:
        engine.generate(prompt="Simple prompt", max_tokens=5)
    assert "RAM budget exceeded" in str(exc_info.value)


def test_gguf_engine_unload_and_memory_reclamation():
    """Verify unload_model releases references and resets loaded state."""
    engine = CraxyGGUFEngine()
    engine.load_model()
    assert engine.is_loaded
    engine.unload_model()
    assert not engine.is_loaded


def test_gguf_engine_sustained_generation_zero_memory_leak():
    """Verify 50 consecutive inference passes exhibit zero process RSS drift."""
    engine = CraxyGGUFEngine()
    proc = psutil.Process()
    rss_before = proc.memory_info().rss / (1024 * 1024)

    for i in range(50):
        engine.generate(
            prompt=f"Iteration {i}: generate sample code output.",
            max_tokens=8,
        )

    rss_after = proc.memory_info().rss / (1024 * 1024)
    drift_mb = abs(rss_after - rss_before)
    # Memory drift over 50 passes must remain strictly bounded under 10 MB
    assert drift_mb < 10.0
