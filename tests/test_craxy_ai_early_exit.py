"""
Unit tests for Speculative Layer Skipping and Early Exit Inference (Phase 2 Stage 7).

Validates:
- EarlyExitClassifier memory footprint (< 2.5 MB).
- Gating decision activates on layer 14 for high-confidence / low-entropy states.
- Rejection of early exit on non-exit layers.
- Shannon entropy mathematical bounds [0.0, 1.0].
- Zero memory leakage across 1,000 continuous evaluations.
"""

import gc
import psutil
import os
import torch
import pytest

from src.craxy_ai.core.early_exit import EarlyExitClassifier, EarlyExitDecision
from src.craxy_ai.config import DEFAULT_MODEL_CONFIG


def test_early_exit_memory_footprint_under_ceiling():
    """Verify that early exit linear head consumes < 2.5 MB of RAM."""
    classifier = EarlyExitClassifier()
    assert classifier.memory_footprint_mb < 2.5
    assert classifier.exit_layer == 14
    assert classifier.total_layers == 28


def test_early_exit_bypasses_on_non_exit_layer():
    """Verify that layers other than L_exit never trigger early exit."""
    classifier = EarlyExitClassifier(exit_layer=14)
    state = torch.randn(1, 1, DEFAULT_MODEL_CONFIG.hidden_dim)

    # Layer 5 should not exit
    dec_5 = classifier.evaluate_state(state, current_layer=5)
    assert not dec_5.should_exit
    assert dec_5.latency_saved_pct == 0.0

    # Layer 27 should not exit
    dec_27 = classifier.evaluate_state(state, current_layer=27)
    assert not dec_27.should_exit


def test_early_exit_activates_on_peaked_activation():
    """Verify that a highly aligned activation triggers early exit with 50% latency saved."""
    classifier = EarlyExitClassifier(exit_layer=14, confidence_threshold=0.50)
    
    # Construct an activation vector aligned with first basis row of exit_head
    aligned_state = classifier.exit_head.weight[0].unsqueeze(0).unsqueeze(0) * 10.0

    decision: EarlyExitDecision = classifier.evaluate_state(aligned_state, current_layer=14)

    assert decision.should_exit is True
    assert decision.exit_layer == 14
    assert decision.confidence >= 0.50
    assert decision.latency_saved_pct == 50.0  # (28 - 14) / 28 = 50.0%


def test_early_exit_entropy_bounds():
    """Verify normalized Shannon entropy calculation bounds [0.0, 1.0]."""
    classifier = EarlyExitClassifier()

    # Uniform distribution -> max entropy ~1.0
    uniform_probs = torch.ones(256) / 256.0
    entropy_uniform = classifier.calculate_entropy(uniform_probs)
    assert 0.95 <= entropy_uniform <= 1.05

    # Peaked one-hot distribution -> min entropy ~0.0
    peaked_probs = torch.zeros(256)
    peaked_probs[0] = 1.0
    entropy_peaked = classifier.calculate_entropy(peaked_probs)
    assert 0.0 <= entropy_peaked <= 0.05


def test_early_exit_zero_memory_leakage_over_1000_evals():
    """Verify zero memory leakage across 1,000 evaluations."""
    classifier = EarlyExitClassifier()
    state = torch.randn(1, 1, DEFAULT_MODEL_CONFIG.hidden_dim)
    gc.collect()
    rss_initial = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

    for _ in range(1000):
        classifier.evaluate_state(state, current_layer=14)

    gc.collect()
    rss_final = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    drift = rss_final - rss_initial

    assert drift < 5.0  # Drift < 5 MB
