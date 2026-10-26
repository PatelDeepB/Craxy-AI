"""Data classes and exception types for CraxyPagedKV cache engine."""

from dataclasses import dataclass


class PagedKVCacheError(Exception):
    """Base exception for errors in CraxyPagedKVCache operations."""
    pass


class BlockOutOfBoundsError(PagedKVCacheError):
    """Raised when an operation requests a block index beyond allocated capacity."""
    pass


class CacheStorageIOError(PagedKVCacheError):
    """Raised when disk backing read or write fails."""
    pass


@dataclass(frozen=True)
class PagedKVConfig:
    """Configuration parameters for two-tier paged KV cache."""
    max_tokens: int = 131072
    block_size_tokens: int = 64
    ram_window_tokens: int = 2048
    sink_tokens: int = 64
    bytes_per_element: int = 1  # INT8 quantization

    def __post_init__(self) -> None:
        if self.ram_window_tokens <= self.sink_tokens:
            raise ValueError("ram_window_tokens must be greater than sink_tokens.")
        if self.ram_window_tokens % self.block_size_tokens != 0:
            raise ValueError("ram_window_tokens must be a multiple of block_size_tokens.")
        if self.max_tokens % self.block_size_tokens != 0:
            raise ValueError("max_tokens must be a multiple of block_size_tokens.")
