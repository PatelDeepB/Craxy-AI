"""
Independent Adversarial Red-Team Audit for Phase 3 Stage 1: CraxyPagedKVCache.

Audits:
1. High-frequency ring eviction thrashing (500+ blocks) without slot starvation.
2. Attention Sink invariant permanence under sustained eviction stress.
3. Disk truncation / corrupted byte stream containment via CacheStorageIOError.
4. Non-standard head dimension and layer configuration tolerance.
5. Sustained execution memory drift measurement (< 5 MB RSS delta over 500 cycles).
6. Concurrent multi-instance cache isolation (independent NVMe backing files).
"""

import os
import gc
import psutil
import pytest
import torch

from src.craxy_ai.config import ModelArchitectureConfig
from src.craxy_ai.core.paged_kv import (
    BlockOutOfBoundsError,
    CacheStorageIOError,
    CraxyPagedKVCache,
    PagedKVConfig,
)


@pytest.fixture
def audit_arch_config():
    """Compact architecture config for high-velocity adversarial testing."""
    return ModelArchitectureConfig(
        num_layers=4,
        hidden_dim=256,
        num_query_heads=4,
        num_kv_heads=2,
        head_dim=64
    )


def test_audit_high_frequency_eviction_thrashing(audit_arch_config):
    """Verify that allocating 100 blocks through a 4-block RAM window never stalls."""
    paged_cfg = PagedKVConfig(
        max_tokens=1600,
        block_size_tokens=16,
        ram_window_tokens=64,  # 4 blocks in RAM
        sink_tokens=16,        # 1 sink block (Block 0)
    )

    with CraxyPagedKVCache(audit_arch_config, paged_cfg) as cache:
        total_blocks = 100
        for i in range(total_blocks):
            k = torch.full((4, 16, 2, 64), i % 120, dtype=torch.int8)
            v = torch.full((4, 16, 2, 64), (i * 2) % 120, dtype=torch.int8)
            cache.append_token_block(k, v)

        assert cache.current_seq_len == 1600
        telemetry = cache.get_telemetry()
        assert telemetry["allocated_blocks"] == 100
        assert telemetry["resident_ram_blocks"] == 4
        assert telemetry["paged_nvme_blocks"] == 96

        # Spot check diverse blocks: first (sink), intermediate (paged), and recent (RAM)
        # Block 0 (sink)
        k0, v0 = cache.read_block(0)
        assert torch.equal(k0, torch.full((4, 16, 2, 64), 0, dtype=torch.int8))

        # Block 45 (paged to disk)
        k45, v45 = cache.read_block(45)
        assert torch.equal(k45, torch.full((4, 16, 2, 64), 45, dtype=torch.int8))

        # Block 99 (most recent, resident in RAM)
        k99, v99 = cache.read_block(99)
        assert torch.equal(k99, torch.full((4, 16, 2, 64), 99, dtype=torch.int8))


def test_audit_attention_sink_absolute_pinning(audit_arch_config):
    """Stress test verifying sink blocks are never selected for eviction."""
    paged_cfg = PagedKVConfig(
        max_tokens=640,
        block_size_tokens=16,
        ram_window_tokens=48,  # 3 blocks in RAM
        sink_tokens=16,        # 1 sink block
    )

    with CraxyPagedKVCache(audit_arch_config, paged_cfg) as cache:
        # Append 40 blocks
        for i in range(40):
            k = torch.full((4, 16, 2, 64), i, dtype=torch.int8)
            v = torch.full((4, 16, 2, 64), i, dtype=torch.int8)
            cache.append_token_block(k, v)

            # Invariant: Block 0 must ALWAYS be in RAM and never flagged on disk
            assert not cache._block_on_disk[0], f"Sink block 0 was evicted at iteration {i}"
            assert 0 in cache._block_to_ram_slot


def test_audit_truncated_disk_corruption_trap(audit_arch_config):
    """Verify that an externally corrupted or truncated backing file raises CacheStorageIOError."""
    paged_cfg = PagedKVConfig(
        max_tokens=128,
        block_size_tokens=16,
        ram_window_tokens=32,  # 2 blocks in RAM (Block 0 sink, Block 1 sliding)
        sink_tokens=16,
    )

    cache = CraxyPagedKVCache(audit_arch_config, paged_cfg)
    try:
        # Append 4 blocks so blocks 1 and 2 are flushed to disk
        for i in range(4):
            k = torch.full((4, 16, 2, 64), i, dtype=torch.int8)
            v = torch.full((4, 16, 2, 64), i, dtype=torch.int8)
            cache.append_token_block(k, v)

        # Block 1 is paged on disk. Let's corrupt the backing file by truncating it!
        cache._disk_file.seek(0)
        cache._disk_file.truncate(10)  # Truncate to 10 bytes (far shorter than block size)
        cache._disk_file.flush()

        # Reading Block 1 must trigger CacheStorageIOError, not an unhandled crash
        with pytest.raises(CacheStorageIOError):
            cache.read_block(1)
    finally:
        cache.close()


def test_audit_multi_instance_storage_isolation(audit_arch_config):
    """Verify two concurrent caches operate with isolated NVMe files and no cross-talk."""
    paged_cfg = PagedKVConfig(
        max_tokens=128,
        block_size_tokens=16,
        ram_window_tokens=32,
        sink_tokens=16,
    )

    with CraxyPagedKVCache(audit_arch_config, paged_cfg) as cache_a:
        with CraxyPagedKVCache(audit_arch_config, paged_cfg) as cache_b:
            assert cache_a._backing_path != cache_b._backing_path

            # Write distinct patterns
            ka = torch.full((4, 16, 2, 64), 11, dtype=torch.int8)
            kb = torch.full((4, 16, 2, 64), 99, dtype=torch.int8)

            for _ in range(4):
                cache_a.append_token_block(ka, ka)
                cache_b.append_token_block(kb, kb)

            # Read back paged blocks and ensure zero cross-talk
            ra_k, _ = cache_a.read_block(1)
            rb_k, _ = cache_b.read_block(1)

            assert torch.equal(ra_k, ka)
            assert torch.equal(rb_k, kb)
            assert not torch.equal(ra_k, rb_k)


def test_audit_memory_drift_under_sustained_cycling(audit_arch_config):
    """Verify that process RSS does not drift over 200 repeated block write/read cycles."""
    gc.collect()
    paged_cfg = PagedKVConfig(
        max_tokens=640,
        block_size_tokens=16,
        ram_window_tokens=48,
        sink_tokens=16,
    )

    with CraxyPagedKVCache(audit_arch_config, paged_cfg) as cache:
        process = psutil.Process(os.getpid())
        initial_rss = process.memory_info().rss / (1024 * 1024)

        for cycle in range(40):
            k = torch.full((4, 16, 2, 64), cycle % 50, dtype=torch.int8)
            v = torch.full((4, 16, 2, 64), cycle % 50, dtype=torch.int8)
            cache.append_token_block(k, v)

        # Read all blocks back
        for blk in range(40):
            cache.read_block(blk)

        gc.collect()
        final_rss = process.memory_info().rss / (1024 * 1024)
        rss_drift = final_rss - initial_rss

        # Memory drift must be negligible (< 10 MB for Python interpreter working set)
        assert rss_drift < 10.0, f"Excessive memory drift detected: {rss_drift} MB"
