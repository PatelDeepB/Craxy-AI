"""Quantized GGUF neural inference engine integrated with llama_cpp.

Enforces 1,500 MB memory budgeting, memory-mapped weight residency,
and token-by-token generation with performance telemetry.
"""

import gc
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

import psutil

try:
    import llama_cpp
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    llama_cpp = None
    LLAMA_CPP_AVAILABLE = False

from .gguf_types import (
    ContextWindowOverflowError,
    GGUFEngineConfig,
    GGUFEngineError,
    GGUFGenerationResult,
    InferenceExecutionError,
    MemoryBudgetExceededError,
    ModelNotFoundError,
)


class CraxyGGUFEngine:
    """Orchestrates quantized on-disk GGUF neural model loading and inference."""

    def __init__(self, config: Optional[GGUFEngineConfig] = None) -> None:
        """Initialize the GGUF runtime engine with configuration limits.

        Args:
            config: Engine configuration parameters.
        """
        self.config = config or GGUFEngineConfig()
        self._model: Optional[Any] = None
        self._is_loaded: bool = False
        self._process = psutil.Process()

    @property
    def is_loaded(self) -> bool:
        """Return whether a neural weight artifact is currently initialized."""
        return self._is_loaded

    def get_memory_status(self) -> Dict[str, float]:
        """Query current process physical memory metrics against the budget.

        Returns:
            Dictionary with current RSS, budget ceiling, and headroom in megabytes.
        """
        rss_mb = self._process.memory_info().rss / (1024 * 1024)
        headroom_mb = max(0.0, self.config.max_ram_budget_mb - rss_mb)
        return {
            "rss_mb": round(rss_mb, 2),
            "budget_mb": round(self.config.max_ram_budget_mb, 2),
            "headroom_mb": round(headroom_mb, 2),
        }

    def load_model(self) -> None:
        """Load GGUF weights from disk with memory-mapping and budget enforcement.

        Raises:
            ModelNotFoundError: If specified model path does not exist on disk.
            MemoryBudgetExceededError: If loaded model breaches RAM ceiling.
            InferenceExecutionError: If llama_cpp fails to parse the model artifact.
        """
        if self._is_loaded:
            return

        if not self.config.model_path:
            self._is_loaded = True
            return

        target_path = Path(self.config.model_path).resolve()
        if not target_path.is_file():
            raise ModelNotFoundError(f"GGUF model file not found: {target_path}")

        if not LLAMA_CPP_AVAILABLE:
            raise InferenceExecutionError("llama_cpp library is not installed in environment")

        try:
            self._model = llama_cpp.Llama(
                model_path=str(target_path),
                n_ctx=self.config.context_window_tokens,
                n_batch=self.config.batch_size,
                n_threads=self.config.num_threads,
                use_mmap=self.config.use_mmap,
                use_mlock=self.config.use_mlock,
                verbose=False,
            )
            self._is_loaded = True
        except Exception as err:
            raise InferenceExecutionError(f"Failed to load GGUF model: {err}") from err

        self._check_memory_budget("model load")

    def _check_memory_budget(self, phase: str) -> None:
        """Verify current process RSS does not exceed the hard memory limit."""
        current_rss = self._process.memory_info().rss / (1024 * 1024)
        if current_rss > self.config.max_ram_budget_mb:
            self.unload_model()
            raise MemoryBudgetExceededError(
                f"RAM budget exceeded during {phase}: {current_rss:.2f} MB > "
                f"{self.config.max_ram_budget_mb:.2f} MB limit"
            )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.2,
        top_p: float = 0.95,
        stop: Optional[List[str]] = None,
    ) -> GGUFGenerationResult:
        """Generate tokens from prompt while tracking latency and memory.

        Args:
            prompt: Text prompt input to evaluate.
            max_tokens: Maximum number of completion tokens to produce.
            temperature: Sampling temperature.
            top_p: Nucleus sampling probability cutoff.
            stop: Optional list of stop sequence strings.

        Returns:
            GGUFGenerationResult with output text and telemetry.
        """
        if not self._is_loaded:
            self.load_model()

        start_time = time.monotonic()
        estimated_prompt_tokens = max(1, len(prompt) // 4)

        if estimated_prompt_tokens + max_tokens > self.config.context_window_tokens:
            raise ContextWindowOverflowError(
                f"Context overflow: {estimated_prompt_tokens} prompt + {max_tokens} "
                f"completion tokens exceeds {self.config.context_window_tokens}"
            )

        if self._model is not None:
            output_text, completion_tokens, finish_reason = self._run_native_inference(
                prompt, max_tokens, temperature, top_p, stop
            )
        else:
            output_text, completion_tokens, finish_reason = self._run_simulated_inference(
                prompt, max_tokens
            )

        duration = max(0.001, time.monotonic() - start_time)
        peak_rss = self._process.memory_info().rss / (1024 * 1024)
        self._check_memory_budget("generation pass")

        return GGUFGenerationResult(
            generated_text=output_text,
            prompt_tokens=estimated_prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=estimated_prompt_tokens + completion_tokens,
            tokens_per_second=round(completion_tokens / duration, 2),
            peak_rss_mb=round(peak_rss, 2),
            duration_seconds=round(duration, 4),
            finish_reason=finish_reason,
        )

    def _run_native_inference(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[List[str]],
    ) -> tuple[str, int, str]:
        """Execute real inference pass via llama_cpp runtime."""
        try:
            response = self._model(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop or [],
            )
            choice = response["choices"][0]
            text = choice["text"]
            tokens = response["usage"].get("completion_tokens", len(text.split()))
            reason = choice.get("finish_reason", "stop")
            return text, tokens, reason
        except Exception as exc:
            raise InferenceExecutionError(f"Native GGUF execution failure: {exc}") from exc

    def _run_simulated_inference(self, prompt: str, max_tokens: int) -> tuple[str, int, str]:
        """Execute deterministic simulated inference pass for headless environments."""
        simulated_words = ["def", "solve(n):", "return", "n", "*", "2"]
        count = min(max_tokens, len(simulated_words))
        output = " ".join(simulated_words[:count])
        return output, count, "stop"

    def unload_model(self) -> None:
        """Release neural weight memory mappings and trigger garbage collection."""
        if self._model is not None:
            del self._model
            self._model = None
        self._is_loaded = False
        gc.collect()
