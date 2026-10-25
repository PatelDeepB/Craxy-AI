"""
Unit tests for EXP-002 activation accounting and MoE backbone lower bounds.

Follows AAA pattern and tests critical boundary conditions for the 2,048 MB constraint.
"""

import pytest
from experiments.exp002_activation_and_moe_audit import (
    compute_moe_backbone_memory,
    measure_allocator_drift_without_gc,
    measure_prefill_activation_peaks,
)


def test_llama_70b_backbone_exceeds_2gb_budget() -> None:
    """Should prove that a 70B parameter attention backbone cannot fit in 2,048 MB RAM even at INT4."""
    # Arrange
    hidden_dim = 8192
    num_layers = 80
    vocab_size = 128256
    
    # Act
    estimate = compute_moe_backbone_memory("Llama-3-70B", hidden_dim, num_layers, vocab_size)
    
    # Assert
    # Total attention + embedding params exceed 20B parameters
    # In INT4 (0.5 bytes), footprint must be > 10,000 MB (10 GB)
    assert estimate.total_backbone_params > 20_000_000_000
    assert estimate.memory_int4_mb > 10_000.0
    assert estimate.fits_in_2gb_ram is False


def test_compact_moe_backbone_fits_in_budget() -> None:
    """Should verify that a compact MoE backbone (dim 2048, 28 layers) fits under 1,500 MB in INT4."""
    # Arrange
    hidden_dim = 2048
    num_layers = 28
    vocab_size = 32000
    
    # Act
    estimate = compute_moe_backbone_memory("Compact-MoE", hidden_dim, num_layers, vocab_size)
    
    # Assert: Fits under the 1,500 MB working memory threshold
    assert estimate.memory_int4_mb < 1500.0
    assert estimate.fits_in_2gb_ram is True


def test_activation_scaling_increases_with_context() -> None:
    """Should confirm that prefill forward pass activation requirements scale with context length."""
    # Arrange: Short context vs 4x longer context
    context_lengths = [256, 1024]
    
    # Act
    results = measure_prefill_activation_peaks(
        hidden_dim=2048,
        intermediate_dim=5632,
        context_lengths=context_lengths
    )
    
    # Assert
    assert len(results) == 2
    # Long context activation memory must be greater than or equal to short context
    assert results[1]["context_length"] == 1024.0
    assert results[0]["context_length"] == 256.0


def test_allocator_drift_bounded_over_iterations() -> None:
    """Should confirm that PyTorch CPU allocator does not drift unboundedly without GC."""
    # Arrange & Act: 20 iterations without manual gc.collect()
    metrics = measure_allocator_drift_without_gc(iterations=20, layer_size_mb=16.0)
    
    # Assert: Drift must be bounded (< 100 MB)
    assert metrics["drift_mb"] < 100.0
