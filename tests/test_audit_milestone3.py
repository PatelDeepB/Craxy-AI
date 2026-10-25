"""
Independent Red-Team Audit of Milestone 3 In-RAM Neural Reasoning Core (Turn 12).

Audits:
- src.craxy_ai.core.model (KVCacheBuffer, CraxyNeuralCore, INT4 footprint, syntax tag parsing)
- src.craxy_ai.core.prefill (ChunkedPrefillEngine, strict chunk <= 512 enforcement, long sequence stability)
"""

import pytest
import torch

from src.craxy_ai.config import DEFAULT_MODEL_CONFIG, SYSTEM_LIMITS
from src.craxy_ai.core.model import CraxyNeuralCore, KVCacheBuffer
from src.craxy_ai.core.prefill import ChunkedPrefillEngine


def test_kv_cache_invariant_memory_under_extreme_token_accumulation():
    """Verify that accumulating tokens far beyond 2,048 never inflates the INT8 KV cache footprint beyond 112.0 MB."""
    kv_cache = KVCacheBuffer(config=DEFAULT_MODEL_CONFIG, max_tokens=2048)
    initial_footprint = kv_cache.memory_footprint_mb

    assert initial_footprint == 112.0, f"Expected 112.0 MB INT8 KV cache, got {initial_footprint}"

    # Push 10,000 tokens across multiple turns
    for _ in range(20):
        kv_cache.append_kv(layer_idx=0, num_new_tokens=500)
        assert kv_cache.current_seq_len <= 2048
        # Footprint must remain strictly invariant (pre-allocated static INT8 buffer)
        assert kv_cache.memory_footprint_mb == 112.0


def test_chunked_prefill_strict_chunk_partition_and_ceiling():
    """Verify that a 4,096-token sequence is strictly partitioned into chunks <= 512 tokens."""
    core = CraxyNeuralCore()
    prefill_engine = ChunkedPrefillEngine(core, chunk_limit=512)

    # Ingest 4,096 tokens
    res = prefill_engine.ingest_context(total_seq_len=4096)

    # 4096 / 512 = 8 chunks
    assert res.num_chunks == 8
    assert res.total_tokens == 4096
    assert res.fits_under_ceiling
    # Activation delta must be strictly capped (measured under 30 MB)
    assert res.activation_delta_mb <= 30.0


def test_int4_backbone_weight_budget_adherence():
    """Verify that INT4 quantized backbone weights strictly occupy <= 768.0 MB."""
    core = CraxyNeuralCore()
    
    # INT4 uses 0.5 bytes per parameter
    assert core.int4_footprint_mb <= 768.0
    assert core.int4_footprint_mb < 750.0  # Exactly ~734 MB


def test_syntax_tag_parser_adversarial_and_malformed_inputs():
    """Verify syntax tag parser resilience against unclosed, empty, and multiple tags."""
    core = CraxyNeuralCore()

    # Empty tag
    is_tool, tool_type, payload = core.parse_syntax_tags("<<retrieve: >>")
    assert is_tool
    assert tool_type == "retrieve"
    assert payload == ""

    # Unclosed tag (must not trigger tool invocation)
    is_tool_unclosed, _, _ = core.parse_syntax_tags("<<calc: 15 * 30")
    assert not is_tool_unclosed

    # Multiple tags (must parse first cleanly without error)
    text = "Let's check <<retrieve: 'capital of France'>> and then <<calc: '10 + 20'>>"
    is_tool_multi, t_type, p_load = core.parse_syntax_tags(text)
    assert is_tool_multi
    assert t_type == "retrieve"
    assert p_load == "capital of France"
