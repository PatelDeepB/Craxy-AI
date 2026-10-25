"""
Unit tests for EXP-005 blueprint edge-case evaluations.

Follows AAA pattern and tests critical boundary conditions for relevance filtering,
syntax resilience, and context window limits.
"""

import pytest
from experiments.exp005_master_blueprint_redteam import (
    apply_relevance_filtering_gate,
    handle_tool_call_with_fallback,
    manage_rolling_context_window,
)


def test_relevance_filtering_gate_accepts_high_confidence_rejects_distractors() -> None:
    """Should accept documents >= threshold (0.65) and reject irrelevant distractors."""
    # Arrange
    candidates = [
        (1, "Highly relevant medical fact", 0.85),
        (2, "Borderline document", 0.65),
        (3, "Completely unrelated noise", 0.35)
    ]
    
    # Act
    results = apply_relevance_filtering_gate(candidates, similarity_threshold=0.65)
    
    # Assert
    assert len(results) == 3
    assert results[0].accepted_by_gate is True
    assert results[1].accepted_by_gate is True
    assert results[2].accepted_by_gate is False


def test_tool_syntax_resilience_success_case() -> None:
    """Should return clean SUCCESS and INJECT_RESULT on valid arithmetic."""
    # Arrange
    valid_expr = "150 * (24 + 18)"
    
    # Act
    res = handle_tool_call_with_fallback(valid_expr)
    
    # Assert
    assert res["status"] == "SUCCESS"
    assert res["computed_value"] == 6300.0
    assert res["recovery_action"] == "INJECT_RESULT"


def test_tool_syntax_resilience_syntax_error_recovery() -> None:
    """Should trap syntax error and produce PROMPT_SYNTAX_REPAIR recovery action."""
    # Arrange: Unbalanced parenthesis
    malformed_expr = "150 * (24 + 18"
    
    # Act
    res = handle_tool_call_with_fallback(malformed_expr)
    
    # Assert
    assert res["status"] == "ERROR_RECOVERED"
    assert res["computed_value"] is None
    assert res["recovery_action"] == "PROMPT_SYNTAX_REPAIR"


def test_rolling_context_window_enforces_ceiling() -> None:
    """Should truncate history turns to strictly guarantee KV cache stays under 2,048 tokens."""
    # Arrange: 10 turns of 250 tokens each (2,500 history tokens) + 512 context + 150 prompt
    history = [250] * 10
    system_tokens = 150
    retrieved_tokens = 512
    max_tokens = 2048
    
    # Act
    state = manage_rolling_context_window(
        system_tokens=system_tokens,
        retrieved_tokens=retrieved_tokens,
        history_turns=history,
        max_token_ceiling=max_tokens
    )
    
    # Assert
    assert state.total_tokens <= max_tokens
    assert state.was_truncated is True
    # KV Cache in INT8 for <= 2048 tokens must be <= 115 MB
    assert state.kv_cache_int8_mb <= 115.0
