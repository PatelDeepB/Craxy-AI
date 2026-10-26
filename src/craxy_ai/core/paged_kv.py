"""NVMe-Paged 128K Key-Value Cache Engine (CraxyPagedKV). Breaks 128K context memory wall within 2,048 MB RAM ceiling."""

import math
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.craxy_ai.config import DEFAULT_MODEL_CONFIG
from src.craxy_ai.core.paged_kv_types import (
    BlockOutOfBoundsError,
    CacheStorageIOError,
    PagedKVCacheError,
    PagedKVConfig,
)


class CraxyPagedKVCache:
    """
    Two-tier paged Key-Value cache supporting up to 128K context tokens.

    Pins attention sinks and recent window in RAM, paging older blocks to NVMe.
    """

    def __init__(
        self,
        arch_config: Any = DEFAULT_MODEL_CONFIG,
        paged_config: Optional[PagedKVConfig] = None,
        backing_dir: Optional[str] = None
    ) -> None:
        if isinstance(arch_config, PagedKVConfig):
            self.cfg = arch_config
            self.arch = DEFAULT_MODEL_CONFIG
        else:
            self.arch = arch_config
            self.cfg = paged_config or PagedKVConfig()
        
        # Dimensions per block
        # Shape per block: (num_layers, 2 [K/V], block_size, num_kv_heads, head_dim)
        self.block_elements = (
            self.arch.num_layers * 2 * self.cfg.block_size_tokens * 
            self.arch.num_kv_heads * self.arch.head_dim
        )
        self.bytes_per_block = self.block_elements * self.cfg.bytes_per_element
        self.total_blocks = self.cfg.max_tokens // self.cfg.block_size_tokens
        self.ram_blocks_count = self.cfg.ram_window_tokens // self.cfg.block_size_tokens
        self.sink_blocks_count = max(1, self.cfg.sink_tokens // self.cfg.block_size_tokens)

        # In-RAM tensor allocation: (ram_blocks, layers, 2, block_size, heads, dim)
        self._ram_buffer = torch.zeros(
            (self.ram_blocks_count, self.arch.num_layers, 2, 
             self.cfg.block_size_tokens, self.arch.num_kv_heads, self.arch.head_dim),
            dtype=torch.int8
        )

        # Block mapping structures
        self._current_tokens = 0
        self._block_on_disk: List[bool] = [False] * self.total_blocks
        self._block_to_ram_slot: Dict[int, int] = {}
        self._ram_slot_to_block: Dict[int, int] = {}
        self._disk_file: Optional[Any] = None
        self._backing_path: Optional[str] = None

        self._init_storage(backing_dir)

    def _init_storage(self, backing_dir: Optional[str]) -> None:
        """Initialize backing NVMe file for paged blocks."""
        if backing_dir and not os.path.exists(backing_dir):
            os.makedirs(backing_dir, exist_ok=True)
            
        target_dir = backing_dir or tempfile.gettempdir()
        self._backing_path = os.path.join(target_dir, f"craxy_paged_kv_{os.getpid()}_{id(self)}.bin")
        try:
            self._disk_file = open(self._backing_path, "w+b")
        except OSError as exc:
            raise CacheStorageIOError(f"Failed to open paged KV backing file: {exc}") from exc

    @property
    def current_seq_len(self) -> int:
        """Return currently ingested sequence length in tokens."""
        return self._current_tokens

    @property
    def active_token_count(self) -> int:
        """Alias property for current_seq_len."""
        return self._current_tokens

    @property
    def ram_footprint_mb(self) -> float:
        """Return physical RAM consumed by the active working buffer in MB."""
        return self._ram_buffer.nelement() / (1024 * 1024)

    @property
    def disk_footprint_mb(self) -> float:
        """Return storage bytes occupied by paged blocks on disk in MB."""
        paged_count = sum(1 for is_paged in self._block_on_disk if is_paged)
        return (paged_count * self.bytes_per_block) / (1024 * 1024)

    def append_token_block(
        self,
        block_keys: torch.Tensor,
        block_values: torch.Tensor
    ) -> int:
        """
        Append a full block of KV states (block_size tokens) to the paged cache.

        Evicts oldest sliding-window block to NVMe if RAM slots are fully occupied.
        """
        if self._current_tokens + self.cfg.block_size_tokens > self.cfg.max_tokens:
            raise BlockOutOfBoundsError(
                f"Cannot exceed max context of {self.cfg.max_tokens} tokens."
            )

        block_idx = self._current_tokens // self.cfg.block_size_tokens
        slot = self._allocate_ram_slot(block_idx)

        # Store keys and values into allocated RAM slot
        self._ram_buffer[slot, :, 0, :, :, :] = block_keys
        self._ram_buffer[slot, :, 1, :, :, :] = block_values

        self._block_to_ram_slot[block_idx] = slot
        self._ram_slot_to_block[slot] = block_idx
        self._current_tokens += self.cfg.block_size_tokens
        return self._current_tokens

    def append_block(
        self,
        block_keys: Any = None,
        block_values: Any = None
    ) -> int:
        """
        Convenience wrapper to append a block of KV states, accepting numpy or torch arrays.
        Expands 3D (tokens, heads, dim) to 4D (layers, tokens, heads, dim) if necessary.
        """
        if block_keys is None:
            block_keys = torch.zeros(
                (self.arch.num_layers, self.cfg.block_size_tokens, self.arch.num_kv_heads, self.arch.head_dim),
                dtype=torch.int8
            )
        elif isinstance(block_keys, np.ndarray):
            block_keys = torch.from_numpy(block_keys)

        if block_values is None:
            block_values = torch.zeros(
                (self.arch.num_layers, self.cfg.block_size_tokens, self.arch.num_kv_heads, self.arch.head_dim),
                dtype=torch.int8
            )
        elif isinstance(block_values, np.ndarray):
            block_values = torch.from_numpy(block_values)

        if block_keys.dim() == 3:
            block_keys = block_keys.unsqueeze(0).expand(
                self.arch.num_layers, -1, -1, -1
            ).clone()
        if block_values.dim() == 3:
            block_values = block_values.unsqueeze(0).expand(
                self.arch.num_layers, -1, -1, -1
            ).clone()

        if block_keys.dtype != torch.int8:
            block_keys = block_keys.to(torch.int8)
        if block_values.dtype != torch.int8:
            block_values = block_values.to(torch.int8)

        return self.append_token_block(block_keys, block_values)

    def _allocate_ram_slot(self, target_block_idx: int) -> int:
        """Find free RAM slot or evict the oldest non-sink block to NVMe."""
        # Check if an unused slot exists
        for slot in range(self.ram_blocks_count):
            if slot not in self._ram_slot_to_block:
                return slot

        # Evict oldest non-sink slot to disk
        evict_slot = self._find_eviction_candidate()
        resident_block = self._ram_slot_to_block[evict_slot]
        self._flush_block_to_disk(resident_block, evict_slot)

        # Reassign slot
        del self._block_to_ram_slot[resident_block]
        del self._ram_slot_to_block[evict_slot]
        return evict_slot

    def _find_eviction_candidate(self) -> int:
        """Identify slot with oldest block that is not an attention sink."""
        oldest_block = math.inf
        chosen_slot = -1
        for slot, blk in self._ram_slot_to_block.items():
            if blk >= self.sink_blocks_count and blk < oldest_block:
                oldest_block = blk
                chosen_slot = slot

        if chosen_slot == -1:
            raise CacheStorageIOError("No eligible eviction slot found outside sinks.")
        return chosen_slot

    def _flush_block_to_disk(self, block_idx: int, slot: int) -> None:
        """Write block tensor from RAM slot into sector-aligned disk position."""
        if self._disk_file is None:
            raise CacheStorageIOError("Backing disk file is not open.")

        offset = block_idx * self.bytes_per_block
        tensor_bytes = self._ram_buffer[slot].numpy().tobytes()
        try:
            self._disk_file.seek(offset)
            self._disk_file.write(tensor_bytes)
            self._disk_file.flush()
            self._block_on_disk[block_idx] = True
        except OSError as exc:
            raise CacheStorageIOError(f"Error writing block {block_idx} to NVMe: {exc}") from exc

    def read_block(self, block_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve K and V tensors for any block (0 to total allocated blocks).

        Returns tensors from RAM if resident, or fetches from NVMe store.
        """
        max_allocated = self._current_tokens // self.cfg.block_size_tokens
        if block_idx < 0 or block_idx >= max_allocated:
            raise BlockOutOfBoundsError(
                f"Block index {block_idx} out of valid range [0, {max_allocated - 1}]."
            )

        if block_idx in self._block_to_ram_slot:
            slot = self._block_to_ram_slot[block_idx]
            k_tensor = self._ram_buffer[slot, :, 0, :, :, :].clone()
            v_tensor = self._ram_buffer[slot, :, 1, :, :, :].clone()
            return k_tensor, v_tensor

        return self._read_block_from_disk(block_idx)

    def _read_block_from_disk(self, block_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fetch evicted block bytes from NVMe backing store into tensor copy."""
        if not self._block_on_disk[block_idx] or self._disk_file is None:
            raise CacheStorageIOError(f"Block {block_idx} is not available on disk.")

        offset = block_idx * self.bytes_per_block
        try:
            self._disk_file.seek(offset)
            raw_bytes = self._disk_file.read(self.bytes_per_block)
            if len(raw_bytes) != self.bytes_per_block:
                raise CacheStorageIOError(f"Incomplete block read for block {block_idx}.")
            
            block_np = torch.from_numpy(np.frombuffer(raw_bytes, dtype=np.int8).copy())
            reshaped = block_np.reshape(
                self.arch.num_layers, 2, self.cfg.block_size_tokens, 
                self.arch.num_kv_heads, self.arch.head_dim
            )
            k_tensor = reshaped[:, 0, :, :, :].clone()
            v_tensor = reshaped[:, 1, :, :, :].clone()
            return k_tensor, v_tensor
        except OSError as exc:
            raise CacheStorageIOError(f"Error reading block {block_idx} from NVMe: {exc}") from exc

    def get_telemetry(self) -> Dict[str, Any]:
        """Return operational metrics for memory accounting and auditing."""
        paged_count = sum(1 for is_paged in self._block_on_disk if is_paged)
        return {
            "current_tokens": self._current_tokens,
            "max_tokens": self.cfg.max_tokens,
            "allocated_blocks": self._current_tokens // self.cfg.block_size_tokens,
            "resident_ram_blocks": len(self._block_to_ram_slot),
            "paged_nvme_blocks": paged_count,
            "ram_footprint_mb": round(self.ram_footprint_mb, 2),
            "disk_footprint_mb": round(self.disk_footprint_mb, 2),
            "backing_path": self._backing_path
        }

    def close(self) -> None:
        """Close disk file and remove backing file to prevent leakage."""
        if self._disk_file is not None and not self._disk_file.closed:
            self._disk_file.close()
            self._disk_file = None

        if self._backing_path and os.path.exists(self._backing_path):
            try:
                os.remove(self._backing_path)
            except OSError:
                pass
            self._backing_path = None

    def __enter__(self) -> "CraxyPagedKVCache":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


__all__ = [
    "CraxyPagedKVCache",
    "PagedKVConfig",
    "PagedKVCacheError",
    "BlockOutOfBoundsError",
    "CacheStorageIOError",
]

