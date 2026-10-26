"""Unit and integration tests for CraxyPagedKVCache engine.

Validates that 128K context scaling operates under strict RAM bounds by
paging older KV blocks to NVMe while preserving attention sinks and recent window.
"""

from pathlib import Path
import tempfile
import psutil
import pytest
import torch

from src.craxy_ai.config import DEFAULT_MODEL_CONFIG, ModelArchitectureConfig
from src.craxy_ai.core.paged_kv import (
    BlockOutOfBoundsError,
    CraxyPagedKVCache,
    PagedKVCacheError,
    PagedKVConfig,
)


@pytest.fixture
def custom_arch() -> ModelArchitectureConfig:
    """Fixture providing a lightweight architecture configuration for fast testing."""
    return ModelArchitectureConfig(
        num_layers=4,
        hidden_dim=256,
        num_query_heads=4,
        num_kv_heads=2,
        head_dim=64,
    )


@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for backing storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def test_paged_kv_initialization(custom_arch: ModelArchitectureConfig, temp_cache_dir: str):
    """Verify clean initialization, RAM allocation, and backing file setup."""
    paged_cfg = PagedKVConfig(
        max_tokens=1024,
        block_size_tokens=64,
        ram_window_tokens=256,
        sink_tokens=64,
    )
    with CraxyPagedKVCache(
        arch_config=custom_arch,
        paged_config=paged_cfg,
        backing_dir=temp_cache_dir,
    ) as cache:
        assert cache.current_seq_len == 0
        telemetry = cache.get_telemetry()
        assert telemetry["current_tokens"] == 0
        assert telemetry["resident_ram_blocks"] == 0
        assert telemetry["paged_nvme_blocks"] == 0
        assert cache.ram_footprint_mb < 20.0


def test_paged_kv_block_append_and_disk_spill(
    custom_arch: ModelArchitectureConfig, temp_cache_dir: str
):
    """Verify that blocks spill to NVMe when RAM window slots are exceeded."""
    paged_cfg = PagedKVConfig(
        max_tokens=512,
        block_size_tokens=64,
        ram_window_tokens=128,  # Only 2 blocks in RAM (1 sink + 1 sliding)
        sink_tokens=64,
    )

    with CraxyPagedKVCache(
        arch_config=custom_arch,
        paged_config=paged_cfg,
        backing_dir=temp_cache_dir,
    ) as cache:
        shape = (
            custom_arch.num_layers,
            paged_cfg.block_size_tokens,
            custom_arch.num_kv_heads,
            custom_arch.head_dim,
        )

        # Append Block 0 (sink block)
        k0 = torch.full(shape, 10, dtype=torch.int8)
        v0 = torch.full(shape, 20, dtype=torch.int8)
        cache.append_token_block(k0, v0)

        # Append Block 1 (window block)
        k1 = torch.full(shape, 30, dtype=torch.int8)
        v1 = torch.full(shape, 40, dtype=torch.int8)
        cache.append_token_block(k1, v1)

        t1 = cache.get_telemetry()
        assert t1["current_tokens"] == 128
        assert t1["resident_ram_blocks"] == 2
        assert t1["paged_nvme_blocks"] == 0

        # Append Block 2: forces eviction of Block 1 (non-sink) to NVMe
        k2 = torch.full(shape, 50, dtype=torch.int8)
        v2 = torch.full(shape, 60, dtype=torch.int8)
        cache.append_token_block(k2, v2)

        t2 = cache.get_telemetry()
        assert t2["current_tokens"] == 192
        assert t2["resident_ram_blocks"] == 2
        assert t2["paged_nvme_blocks"] == 1


def test_paged_kv_read_reconstruction(
    custom_arch: ModelArchitectureConfig, temp_cache_dir: str
):
    """Verify that reading evicted blocks from NVMe restores exact tensor values."""
    paged_cfg = PagedKVConfig(
        max_tokens=512,
        block_size_tokens=64,
        ram_window_tokens=128,
        sink_tokens=64,
    )

    with CraxyPagedKVCache(
        arch_config=custom_arch,
        paged_config=paged_cfg,
        backing_dir=temp_cache_dir,
    ) as cache:
        shape = (
            custom_arch.num_layers,
            paged_cfg.block_size_tokens,
            custom_arch.num_kv_heads,
            custom_arch.head_dim,
        )

        # Append 3 blocks: Block 0 (sink), Block 1 (evicted), Block 2 (resident)
        cache.append_token_block(
            torch.full(shape, 11, dtype=torch.int8),
            torch.full(shape, 22, dtype=torch.int8),
        )
        cache.append_token_block(
            torch.full(shape, 33, dtype=torch.int8),
            torch.full(shape, 44, dtype=torch.int8),
        )
        cache.append_token_block(
            torch.full(shape, 55, dtype=torch.int8),
            torch.full(shape, 66, dtype=torch.int8),
        )

        # Read Block 1 (which was flushed to disk)
        k_disk, v_disk = cache.read_block(1)
        assert torch.all(k_disk == 33)
        assert torch.all(v_disk == 44)

        # Read Block 0 (which remained in RAM sink)
        k_ram, v_ram = cache.read_block(0)
        assert torch.all(k_ram == 11)
        assert torch.all(v_ram == 22)


def test_paged_kv_out_of_bounds_protection(
    custom_arch: ModelArchitectureConfig, temp_cache_dir: str
):
    """Verify bounds protections for invalid block index reads and capacity overflow."""
    paged_cfg = PagedKVConfig(
        max_tokens=128,
        block_size_tokens=64,
        ram_window_tokens=128,
        sink_tokens=64,
    )

    with CraxyPagedKVCache(
        arch_config=custom_arch,
        paged_config=paged_cfg,
        backing_dir=temp_cache_dir,
    ) as cache:
        shape = (
            custom_arch.num_layers,
            paged_cfg.block_size_tokens,
            custom_arch.num_kv_heads,
            custom_arch.head_dim,
        )

        # Cannot read before appending
        with pytest.raises(BlockOutOfBoundsError):
            cache.read_block(0)

        # Append max blocks (2 blocks = 128 tokens)
        cache.append_token_block(
            torch.zeros(shape, dtype=torch.int8),
            torch.zeros(shape, dtype=torch.int8),
        )
        cache.append_token_block(
            torch.zeros(shape, dtype=torch.int8),
            torch.zeros(shape, dtype=torch.int8),
        )

        # Appending 3rd block exceeds capacity
        with pytest.raises(BlockOutOfBoundsError):
            cache.append_token_block(
                torch.zeros(shape, dtype=torch.int8),
                torch.zeros(shape, dtype=torch.int8),
            )


def test_paged_kv_ram_invariance_under_sustained_scaling(
    custom_arch: ModelArchitectureConfig, temp_cache_dir: str
):
    """Verify that process DRAM RSS remains strictly invariant as context scales."""
    paged_cfg = PagedKVConfig(
        max_tokens=2048,
        block_size_tokens=64,
        ram_window_tokens=256,  # 4 blocks in RAM
        sink_tokens=64,
    )

    process = psutil.Process()
    rss_start = process.memory_info().rss / (1024 * 1024)

    with CraxyPagedKVCache(
        arch_config=custom_arch,
        paged_config=paged_cfg,
        backing_dir=temp_cache_dir,
    ) as cache:
        shape = (
            custom_arch.num_layers,
            paged_cfg.block_size_tokens,
            custom_arch.num_kv_heads,
            custom_arch.head_dim,
        )

        # Append 16 blocks (1,024 tokens)
        for i in range(16):
            cache.append_token_block(
                torch.full(shape, i % 127, dtype=torch.int8),
                torch.full(shape, (i + 1) % 127, dtype=torch.int8),
            )

        telemetry = cache.get_telemetry()
        assert telemetry["current_tokens"] == 1024
        assert telemetry["resident_ram_blocks"] == 4
        assert telemetry["paged_nvme_blocks"] == 12
        assert telemetry["ram_footprint_mb"] < 15.0

    rss_end = process.memory_info().rss / (1024 * 1024)
    drift_mb = abs(rss_end - rss_start)
    assert drift_mb < 30.0  # Zero unbounded memory growth
