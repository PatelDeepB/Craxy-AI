"""
Unit tests for memory accounting models and KV cache calculations.

Follows AAA pattern and tests boundary conditions for the 2,048 MB RAM constraint.
"""

import pytest
from experiments.exp001_memory_streaming_benchmark import (
    KVCacheDimension,
    compute_kv_cache_memory_mb,
    measure_runtime_overhead,
)


def test_kv_cache_memory_calculation_fp16() -> None:
    """Should correctly calculate KV cache memory for FP16 precision."""
    # Arrange: 32 layers, 8 KV heads (GQA), head dimension 128, context 2048, FP16 (2 bytes)
    dim = KVCacheDimension(num_layers=32, num_kv_heads=8, head_dim=128, batch_size=1)
    context_length = 2048
    bytes_per_elem = 2.0  # FP16
    
    # Act
    kv_memory_mb = compute_kv_cache_memory_mb(dim, context_length, bytes_per_elem)
    
    # Assert
    # Total elements = 2 * 32 * 1 * 2048 * 8 * 128 = 134,217,728 elements
    # Total bytes = 134,217,728 * 2 = 268,435,456 bytes = 256.0 MB exactly
    assert round(kv_memory_mb, 2) == 256.0


def test_kv_cache_memory_calculation_int8() -> None:
    """Should halve KV cache memory when using INT8 quantization."""
    # Arrange: 32 layers, 8 KV heads, head dimension 128, context 2048, INT8 (1 byte)
    dim = KVCacheDimension(num_layers=32, num_kv_heads=8, head_dim=128, batch_size=1)
    context_length = 2048
    bytes_per_elem = 1.0  # INT8
    
    # Act
    kv_memory_mb = compute_kv_cache_memory_mb(dim, context_length, bytes_per_elem)
    
    # Assert: 128.0 MB
    assert round(kv_memory_mb, 2) == 128.0


def test_runtime_overhead_leaves_positive_budget() -> None:
    """Should verify that the runtime process consumes less than 2,048 MB."""
    # Arrange & Act
    profile = measure_runtime_overhead()
    
    # Assert: Runtime baseline must leave significant headroom for model execution
    assert profile.torch_loaded_rss_mb < 1000.0
    assert profile.available_budget_mb > 1000.0


def test_kv_cache_scaling_with_context() -> None:
    """Should verify linear scaling of KV cache memory with context length."""
    # Arrange
    dim = KVCacheDimension(num_layers=32, num_kv_heads=8, head_dim=128, batch_size=1)
    
    # Act
    mb_1024 = compute_kv_cache_memory_mb(dim, 1024, 2.0)
    mb_2048 = compute_kv_cache_memory_mb(dim, 2048, 2.0)
    mb_4096 = compute_kv_cache_memory_mb(dim, 4096, 2.0)
    
    # Assert
    assert round(mb_2048, 2) == round(mb_1024 * 2, 2)
    assert round(mb_4096, 2) == round(mb_2048 * 2, 2)
