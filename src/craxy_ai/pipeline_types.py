"""Data structures, execution states, and telemetry outputs for the unified pipeline.

Defines the state machine enums and immutable output records produced
by the tri-tier Craxy-AI inference orchestrator.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class PipelineState(str, Enum):
    """Execution states for the tri-tier inference cycle."""

    IDLE = "IDLE"
    PREFILL = "PREFILL"
    GENERATE = "GENERATE"
    TOOL_DISPATCH = "TOOL_DISPATCH"
    FINALIZE = "FINALIZE"


@dataclass(frozen=True)
class PipelineOutput:
    """Telemetry and generated response from the unified pipeline.

    Attributes:
        response: Output text or synthesized answer.
        retrieval_augmented: Whether external knowledge was retrieved and injected.
        retrieved_chunks_count: Number of chunks retrieved from on-disk store.
        tool_calls: Structured list of tool execution results (math, logic, code).
        tokens_generated: Total token count produced during generation pass.
        total_latency_ms: Total end-to-end wall-clock latency in milliseconds.
        peak_rss_mb: Process physical resident set size during query processing.
        state: Final state of the pipeline control loop upon completion.
        context_tokens_used: Number of active context tokens processed.
        model_engine_type: Identifier of active neural engine (parametric or gguf).
    """

    response: str
    retrieval_augmented: bool
    retrieved_chunks_count: int
    tool_calls: List[Dict[str, Any]]
    tokens_generated: int
    total_latency_ms: float
    peak_rss_mb: float
    state: PipelineState
    context_tokens_used: int = 0
    model_engine_type: str = "parametric"
