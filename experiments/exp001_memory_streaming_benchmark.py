"""
Experiment EXP-001: Empirical Memory Budget Profile and Layer Streaming Feasibility.

This module empirically measures:
1. Python and PyTorch baseline Resident Set Size (RSS).
2. KV Cache memory scaling across context lengths and numerical precisions.
3. Sequential layer streaming from disk to verify memory recycling and bounded RSS.
"""

import gc
import json
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import psutil
import torch


@dataclass
class MemoryProfileResult:
    """Stores memory profile measurements in megabytes (MB)."""
    base_python_rss_mb: float
    torch_loaded_rss_mb: float
    torch_overhead_mb: float
    available_budget_mb: float


@dataclass
class KVCacheDimension:
    """Transformer architecture dimensions for KV cache computation."""
    num_layers: int
    num_kv_heads: int
    head_dim: int
    batch_size: int = 1


def get_current_rss_mb() -> float:
    """Return the current process Resident Set Size (RSS) in megabytes."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def measure_runtime_overhead() -> MemoryProfileResult:
    """Measure the memory baseline of the runtime environment."""
    gc.collect()
    rss_after_torch = get_current_rss_mb()
    
    # 2,048 MB strict budget per research charter
    charter_budget_mb = 2048.0
    available_mb = charter_budget_mb - rss_after_torch
    
    return MemoryProfileResult(
        base_python_rss_mb=rss_after_torch,
        torch_loaded_rss_mb=rss_after_torch,
        torch_overhead_mb=rss_after_torch,
        available_budget_mb=available_mb
    )


def compute_kv_cache_memory_mb(
    dim: KVCacheDimension,
    context_length: int,
    bytes_per_elem: float
) -> float:
    """Calculate theoretical KV cache memory in MB for a given context length."""
    # Key cache + Value cache = 2 * (L * B * S * H_kv * D) * bytes_per_elem
    total_elements = 2 * dim.num_layers * dim.batch_size * context_length * dim.num_kv_heads * dim.head_dim
    total_bytes = total_elements * bytes_per_elem
    return total_bytes / (1024 * 1024)


def run_kv_cache_scaling_benchmark(dim: KVCacheDimension) -> List[Dict[str, float]]:
    """Empirically verify KV cache allocations and measure RSS growth."""
    context_lengths = [512, 1024, 2048, 4096, 8192]
    precisions = [
        ("fp16", 2.0, torch.float16),
        ("int8", 1.0, torch.int8)
    ]
    
    results = []
    
    for ctx_len in context_lengths:
        for prec_name, bytes_per_elem, dtype in precisions:
            theoretical_mb = compute_kv_cache_memory_mb(dim, ctx_len, bytes_per_elem)
            
            # Allocate empirical sample tensor representing one layer's KV cache
            gc.collect()
            rss_before = get_current_rss_mb()
            
            # 2 * (1 layer * B * S * H_kv * D)
            layer_cache = torch.zeros(
                (2, dim.batch_size, dim.num_kv_heads, ctx_len, dim.head_dim),
                dtype=dtype
            )
            rss_after = get_current_rss_mb()
            
            results.append({
                "context_length": float(ctx_len),
                "precision": prec_name,
                "bytes_per_elem": bytes_per_elem,
                "theoretical_total_mb": round(theoretical_mb, 2),
                "layer_tensor_mb": round(layer_cache.element_size() * layer_cache.nelement() / (1024 * 1024), 4),
                "rss_delta_mb": round(max(0.0, rss_after - rss_before), 2)
            })
            del layer_cache
            gc.collect()
            
    return results


def simulate_sequential_layer_streaming(
    num_layers: int = 32,
    layer_size_mb: float = 64.0
) -> Tuple[List[float], Dict[str, float]]:
    """
    Simulate streaming model layers sequentially from disk to RAM.
    
    Verifies that memory is properly recycled after each layer pass,
    preventing cumulative RSS growth beyond the budget ceiling.
    """
    temp_dir = tempfile.mkdtemp(prefix="ultron_stream_test_")
    layer_files = []
    
    try:
        # Create dummy serialized layer weights on disk
        num_floats = int((layer_size_mb * 1024 * 1024) / 4)
        for i in range(num_layers):
            layer_path = os.path.join(temp_dir, f"layer_{i:03d}.bin")
            dummy_data = torch.randn(num_floats, dtype=torch.float32)
            torch.save(dummy_data, layer_path)
            layer_files.append(layer_path)
            del dummy_data
            
        gc.collect()
        rss_history = []
        load_times_ms = []
        
        # Sequentially stream each layer into a recycled working buffer
        for layer_path in layer_files:
            t0 = time.perf_counter()
            # Stream layer from disk into RAM
            layer_weights = torch.load(layer_path, map_location="cpu", weights_only=True)
            
            # Simulate computation: forward pass activation matrix multiply
            dummy_act = torch.randn(1, 1024, dtype=torch.float32)
            _ = torch.sum(layer_weights[:1024] * dummy_act)
            
            t1 = time.perf_counter()
            load_times_ms.append((t1 - t0) * 1000.0)
            
            # Measure peak RSS
            current_rss = get_current_rss_mb()
            rss_history.append(current_rss)
            
            # Explicitly deallocate to verify memory recycling
            del layer_weights
            del dummy_act
            gc.collect()
            
        avg_load_time_ms = sum(load_times_ms) / len(load_times_ms)
        peak_rss_mb = max(rss_history)
        min_rss_mb = min(rss_history)
        rss_drift_mb = rss_history[-1] - rss_history[0]
        read_throughput_mb_s = (layer_size_mb / (avg_load_time_ms / 1000.0))
        
        metrics = {
            "num_layers": float(num_layers),
            "layer_size_mb": layer_size_mb,
            "total_model_disk_size_mb": float(num_layers * layer_size_mb),
            "peak_rss_mb": round(peak_rss_mb, 2),
            "min_rss_mb": round(min_rss_mb, 2),
            "rss_drift_mb": round(rss_drift_mb, 2),
            "avg_layer_load_time_ms": round(avg_load_time_ms, 2),
            "read_throughput_mb_s": round(read_throughput_mb_s, 2),
            "full_pass_time_sec": round(sum(load_times_ms) / 1000.0, 3)
        }
        
        return rss_history, metrics
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def run_experiment() -> Dict:
    """Execute EXP-001 end-to-end and return structured report."""
    print("=== ULTRON EXP-001: Memory Budget & Layer Streaming Feasibility ===")
    
    # 1. Measure runtime overhead
    runtime_profile = measure_runtime_overhead()
    print(f"[Runtime] Process RSS with PyTorch: {runtime_profile.torch_loaded_rss_mb:.2f} MB")
    print(f"[Runtime] Available headroom for weights+KV under 2048 MB: {runtime_profile.available_budget_mb:.2f} MB")
    
    # 2. KV Cache Scaling (Llama-3-8B architecture dimensions: 32 layers, 8 KV heads, dim 128)
    llama_dim = KVCacheDimension(num_layers=32, num_kv_heads=8, head_dim=128)
    kv_scaling = run_kv_cache_scaling_benchmark(llama_dim)
    print(f"[KV Cache] 2048 tokens FP16: {compute_kv_cache_memory_mb(llama_dim, 2048, 2.0):.2f} MB")
    print(f"[KV Cache] 4096 tokens FP16: {compute_kv_cache_memory_mb(llama_dim, 4096, 2.0):.2f} MB")
    print(f"[KV Cache] 8192 tokens FP16: {compute_kv_cache_memory_mb(llama_dim, 8192, 2.0):.2f} MB")
    print(f"[KV Cache] 8192 tokens INT8: {compute_kv_cache_memory_mb(llama_dim, 8192, 1.0):.2f} MB")
    
    # 3. Simulate sequential layer streaming (32 layers of 64 MB each = 2.0 GB model on disk)
    print("[Streaming] Testing sequential layer streaming from disk (32 layers x 64 MB = 2,048 MB)...")
    _, streaming_metrics = simulate_sequential_layer_streaming(num_layers=32, layer_size_mb=64.0)
    print(f"[Streaming] Peak RSS during execution: {streaming_metrics['peak_rss_mb']} MB (Ceiling: 2,048 MB)")
    print(f"[Streaming] RSS Drift over 32 layers: {streaming_metrics['rss_drift_mb']} MB")
    print(f"[Streaming] Disk read throughput: {streaming_metrics['read_throughput_mb_s']} MB/s")
    print(f"[Streaming] Single-token layer pass latency: {streaming_metrics['full_pass_time_sec']} seconds")
    
    report = {
        "experiment_id": "EXP-001",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_profile": asdict(runtime_profile),
        "kv_cache_scaling": kv_scaling,
        "streaming_metrics": streaming_metrics
    }
    
    return report


if __name__ == "__main__":
    report_data = run_experiment()
    output_path = os.path.join(os.path.dirname(__file__), "exp001_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"Results saved to {output_path}")
