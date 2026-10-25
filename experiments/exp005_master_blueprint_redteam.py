"""
Experiment EXP-005: Red-Team Validation of Craxy-AI Architectural Master Blueprint.

Conducted by ULTRON 2 to audit edge cases and stress-test the blueprint:
1. Relevance Filtering Gate: Rejection of low-similarity / distracting retrieved contexts.
2. Tool Syntax Drift Resilience: Error recovery on malformed arithmetic expressions.
3. Rolling Context Window Manager: Strict enforcement of the 2,048 token KV cache ceiling.
"""

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.exp004_concurrent_memory_and_tool_audit import execute_sandboxed_math


@dataclass
class RetrievedChunk:
    """Represents a document chunk retrieved from on-disk memory."""
    chunk_id: int
    text: str
    similarity_score: float
    accepted_by_gate: bool


@dataclass
class ContextWindowState:
    """Tracks active tokens and estimated KV cache memory."""
    system_prompt_tokens: int
    retrieved_context_tokens: int
    conversation_tokens: int
    total_tokens: int
    kv_cache_int8_mb: float
    was_truncated: bool


def apply_relevance_filtering_gate(
    retrieved_candidates: List[Tuple[int, str, float]],
    similarity_threshold: float = 0.65
) -> List[RetrievedChunk]:
    """Filter out low-confidence or distracting retrieval chunks before model injection."""
    filtered = []
    for chunk_id, text, score in retrieved_candidates:
        accepted = score >= similarity_threshold
        filtered.append(RetrievedChunk(
            chunk_id=chunk_id,
            text=text,
            similarity_score=round(score, 4),
            accepted_by_gate=accepted
        ))
    return filtered


def handle_tool_call_with_fallback(expression_str: str) -> Dict[str, Any]:
    """Execute tool call with syntax error trapping and structured recovery feedback."""
    execution_result = execute_sandboxed_math(expression_str)
    if execution_result["success"]:
        return {
            "status": "SUCCESS",
            "computed_value": execution_result["result"],
            "recovery_action": "INJECT_RESULT",
            "message": f"Successfully evaluated: {execution_result['result']}"
        }
    
    # Analyze error category for self-correction feedback
    err_str = execution_result["error"] or "UnknownError"
    if "SyntaxError" in err_str:
        recovery = "PROMPT_SYNTAX_REPAIR"
        feedback = "Syntax error in expression. Please check parentheses and operators."
    elif "ZeroDivisionError" in err_str:
        recovery = "FLAG_UNDEFINED_OPERATION"
        feedback = "Division by zero encountered. Expression mathematically undefined."
    elif "OverflowError" in err_str:
        recovery = "FLAG_BOUNDS_OVERFLOW"
        feedback = "Numerical overflow. Exponent exceeds safety bound."
    else:
        recovery = "FALLBACK_TO_CORE_REASONING"
        feedback = f"Calculation rejected: {err_str}"
        
    return {
        "status": "ERROR_RECOVERED",
        "computed_value": None,
        "recovery_action": recovery,
        "message": feedback
    }


def manage_rolling_context_window(
    system_tokens: int,
    retrieved_tokens: int,
    history_turns: List[int],
    max_token_ceiling: int = 2048,
    num_layers: int = 28,
    num_heads: int = 8,
    head_dim: int = 128
) -> ContextWindowState:
    """Enforce strict KV cache ceiling by rolling eviction of oldest conversation turns."""
    base_overhead = system_tokens + retrieved_tokens
    allowed_for_history = max(0, max_token_ceiling - base_overhead)
    
    kept_history_tokens = 0
    was_truncated = False
    
    # Retain newest turns first
    for turn_tokens in reversed(history_turns):
        if kept_history_tokens + turn_tokens <= allowed_for_history:
            kept_history_tokens += turn_tokens
        else:
            was_truncated = True
            
    total_tokens = base_overhead + kept_history_tokens
    
    # INT8 KV Cache bytes: 2 * L * 1 * S * H * D * 1 byte
    kv_cache_bytes = 2 * num_layers * 1 * total_tokens * num_heads * head_dim
    kv_cache_mb = kv_cache_bytes / (1024 * 1024)
    
    return ContextWindowState(
        system_prompt_tokens=system_tokens,
        retrieved_context_tokens=retrieved_tokens,
        conversation_tokens=kept_history_tokens,
        total_tokens=total_tokens,
        kv_cache_int8_mb=round(kv_cache_mb, 2),
        was_truncated=was_truncated
    )


def run_blueprint_redteam_benchmark() -> Dict[str, Any]:
    """Execute EXP-005 edge-case evaluations."""
    # 1. Relevance Gate Evaluation
    mock_candidates = [
        (101, "Exact target fact about Llama-3 architecture.", 0.88),
        (102, "Marginally related paper discussion on MoE.", 0.68),
        (103, "Unrelated sports news article.", 0.42),
        (104, "Misleading distractor text with random keywords.", 0.31)
    ]
    relevance_results = apply_relevance_filtering_gate(mock_candidates, similarity_threshold=0.65)
    
    # 2. Syntax Drift and Malformed Tool Call Scenarios
    syntax_scenarios = [
        "150 * (24 + 18)",           # Valid
        "150 * (24 + 18",            # Missing parenthesis
        "100 / 0",                   # Division by zero
        "2 ** 50000",                # Exponent overflow
        "x + 5",                     # Undefined variable
        "empty_call()"               # Disallowed function call
    ]
    syntax_results = [handle_tool_call_with_fallback(expr) for expr in syntax_scenarios]
    
    # 3. Rolling Context Window Eviction Test
    # Simulating 10 turns of 300 tokens each (3,000 conversation tokens)
    history_turns = [300] * 10
    window_state = manage_rolling_context_window(
        system_tokens=150,
        retrieved_tokens=512,
        history_turns=history_turns,
        max_token_ceiling=2048
    )
    
    report = {
        "experiment_id": "EXP-005",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "relevance_filtering_gate": [asdict(r) for r in relevance_results],
        "tool_syntax_resilience": syntax_results,
        "rolling_context_window_audit": asdict(window_state)
    }
    
    output_path = os.path.join(os.path.dirname(__file__), "exp005_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    return report


if __name__ == "__main__":
    report = run_blueprint_redteam_benchmark()
    print("EXP-005 completed successfully.")
