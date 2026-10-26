"""Data classes, telemetry descriptors, and typed errors for GGUF runtime inference.

Defines configuration settings, generation metrics, and memory bounds
enforced during quantized neural execution within the 2 GB RAM limit.
"""

from dataclasses import dataclass, field
from typing import Optional


class GGUFEngineError(Exception):
    """Base exception for all GGUF inference engine operations."""


class ModelNotFoundError(GGUFEngineError):
    """Raised when the specified GGUF weight artifact cannot be found on disk."""


class MemoryBudgetExceededError(GGUFEngineError):
    """Raised when memory consumption breaches the 1,500 MB model ceiling."""


class ContextWindowOverflowError(GGUFEngineError):
    """Raised when prompt + generation tokens exceed configured context window."""


class InferenceExecutionError(GGUFEngineError):
    """Raised when underlying llama_cpp or runtime execution fails."""


@dataclass(frozen=True)
class GGUFEngineConfig:
    """Configuration parameters for GGUF weight loading and execution.

    Attributes:
        model_path: File system path to the .gguf binary weight artifact.
        context_window_tokens: Maximum addressable context window.
        batch_size: Prompt evaluation chunk / batch size.
        num_threads: Number of CPU threads dedicated to GEMM computation.
        use_mmap: Whether to memory-map weights directly from storage.
        use_mlock: Whether to lock memory pages (default False to conserve RAM).
        max_ram_budget_mb: Hard physical memory threshold for process RSS.
    """

    model_path: Optional[str] = None
    context_window_tokens: int = 2048
    batch_size: int = 512
    num_threads: int = 4
    use_mmap: bool = True
    use_mlock: bool = False
    max_ram_budget_mb: float = 1500.0


@dataclass(frozen=True)
class GGUFGenerationResult:
    """Telemetry captured from a quantized token generation pass.

    Attributes:
        generated_text: Output text decoded from generated tokens.
        prompt_tokens: Number of tokens consumed in prompt evaluation.
        completion_tokens: Number of tokens emitted during generation.
        total_tokens: Aggregate count of prompt and completion tokens.
        tokens_per_second: Generation speed throughput.
        peak_rss_mb: Process physical resident set size during inference.
        duration_seconds: Total wall-clock duration of the inference call.
        finish_reason: Reason generation ceased ('stop', 'length', etc.).
    """

    generated_text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    tokens_per_second: float
    peak_rss_mb: float
    duration_seconds: float
    finish_reason: str = "stop"
