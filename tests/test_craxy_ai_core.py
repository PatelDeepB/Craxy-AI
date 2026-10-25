"""
Unit tests for the Craxy-AI in-RAM neural reasoning core and chunked prefill engine.

Verifies model dimensions, INT8 KV cache footprint (<= 112 MB), chunked prefill
activation surge suppression (<= 35 MB), rolling window eviction, and syntax tag parsing.
"""

import pytest
import torch
from src.craxy_ai.config import DEFAULT_MODEL_CONFIG, SYSTEM_LIMITS
from src.craxy_ai.core.model import CraxyNeuralCore, KVCacheBuffer
from src.craxy_ai.core.prefill import ChunkedPrefillEngine


@pytest.fixture
def neural_core() -> CraxyNeuralCore:
    """Fixture providing a CraxyNeuralCore instance."""
    return CraxyNeuralCore(config=DEFAULT_MODEL_CONFIG)


def test_kv_cache_memory_bounded_under_112mb() -> None:
    """Should verify that the INT8 KV cache consumes <= 112.0 MB for 2,048 tokens."""
    # Arrange & Act
    kv_cache = KVCacheBuffer(config=DEFAULT_MODEL_CONFIG, max_tokens=2048)
    
    # Assert
    # 28 layers * 2 (K & V) * 1 batch * 2048 tokens * 8 heads * 128 dim * 1 byte = 112.0 MB
    assert round(kv_cache.memory_footprint_mb, 2) == 112.0


def test_neural_core_architecture_dimensions(neural_core: CraxyNeuralCore) -> None:
    """Should verify the 1.5B compact parameter count and ~734 MB INT4 footprint."""
    # Arrange & Act
    params = neural_core.total_params
    int4_mb = neural_core.int4_footprint_mb
    
    # Assert
    assert 1.4e9 <= params <= 1.6e9  # 1.5B scale
    assert 700.0 <= int4_mb <= 800.0  # Under 800 MB INT4


def test_chunked_prefill_caps_activation_surge(neural_core: CraxyNeuralCore) -> None:
    """Should verify chunked prefill caps activation memory delta to <= 35 MB."""
    # Arrange: 1,536 tokens ingested in 512-token chunks
    engine = ChunkedPrefillEngine(core=neural_core, chunk_limit=512)
    
    # Act
    result = engine.ingest_context(total_seq_len=1536)
    
    # Assert
    assert result.num_chunks == 3
    assert result.activation_delta_mb <= 35.0
    assert result.fits_under_ceiling is True
    assert result.peak_rss_mb <= SYSTEM_LIMITS.max_ram_rss_mb


def test_syntax_tag_parser_detection(neural_core: CraxyNeuralCore) -> None:
    """Should accurately parse retrieval and calculation tags from text."""
    # Arrange
    text_retrieve = "I need facts about <<retrieve: 'transformer attention'>> in memory."
    text_calc = "The price is <<calc: '120 * 0.85'>> dollars."
    text_plain = "Standard conversational response without tool tags."
    
    # Act
    is_r, t_type_r, payload_r = neural_core.parse_syntax_tags(text_retrieve)
    is_c, t_type_c, payload_c = neural_core.parse_syntax_tags(text_calc)
    is_p, t_type_p, payload_p = neural_core.parse_syntax_tags(text_plain)
    
    # Assert
    assert is_r is True and t_type_r == "retrieve" and payload_r == "transformer attention"
    assert is_c is True and t_type_c == "calc" and payload_c == "120 * 0.85"
    assert is_p is False and t_type_p is None and payload_p is None


def test_rolling_window_context_eviction() -> None:
    """Should enforce context ceiling and reset sequence pointer gracefully."""
    # Arrange
    kv_cache = KVCacheBuffer(config=DEFAULT_MODEL_CONFIG, max_tokens=2048)
    
    # Act: Append beyond 2,048 tokens
    seq_len1 = kv_cache.append_kv(layer_idx=0, num_new_tokens=1500)
    seq_len2 = kv_cache.append_kv(layer_idx=0, num_new_tokens=1000)
    
    # Assert: Should not exceed max_tokens 2048
    assert seq_len1 == 1500
    assert seq_len2 <= 2048
    
    # Act & Assert reset
    kv_cache.reset()
    assert kv_cache.current_seq_len == 0
