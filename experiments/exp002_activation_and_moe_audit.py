"""
Experiment EXP-002: Independent Audit of Activation Peaks, Allocator Drift,
and Mixture-of-Experts (MoE) Backbone Memory Bounds.

Conducted by ULTRON 2 to independently challenge Turn 1 assumptions:
1. Real activation memory during prefill (S tokens) vs decode (1 token).
2. Heap allocator memory drift over 100 consecutive token iterations without manual GC sweeps.
3. Architectural memory lower bounds for MoE attention backbones.
4. Serialized non-contiguous expert disk reads per layer on NVMe.
"""

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
class BackboneMemoryEstimate:
    """Stores calculated parameter count and RAM footprint of shared backbone."""
    architecture_name: str
    hidden_dim: int
    num_layers: int
    vocab_size: int
    embedding_params: int
    attention_params: int
    lm_head_params: int
    total_backbone_params: int
    memory_fp16_mb: float
    memory_int8_mb: float
    memory_int4_mb: float
    fits_in_2gb_ram: bool


def get_current_rss_mb() -> float:
    """Return the current process Resident Set Size (RSS) in megabytes."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def measure_prefill_activation_peaks(
    hidden_dim: int = 4096,
    intermediate_dim: int = 14336,
    context_lengths: List[int] = None
) -> List[Dict[str, float]]:
    """Measure peak process RSS during Transformer layer forward passes across context lengths."""
    if context_lengths is None:
        context_lengths = [512, 1024, 2048, 4096]
        
    results = []
    
    # Layer weights: Q, K, V, O projections + Gate, Up, Down FFN projections
    # Note: On x86_64 CPU without AVX-512 FP16, PyTorch float32 runs native AVX2 vectorization
    # whereas CPU float16 incurs a 170x software emulation penalty.
    w_qkv = torch.randn(hidden_dim, 3 * hidden_dim, dtype=torch.float32)
    w_gate = torch.randn(hidden_dim, intermediate_dim, dtype=torch.float32)
    w_up = torch.randn(hidden_dim, intermediate_dim, dtype=torch.float32)
    
    for ctx_len in context_lengths:
        rss_start = get_current_rss_mb()
        
        # Forward prefill tensor
        x = torch.randn(1, ctx_len, hidden_dim, dtype=torch.float32)
        
        # Linear projections
        qkv = torch.matmul(x, w_qkv)
        gate = torch.matmul(x, w_gate)
        up = torch.matmul(x, w_up)
        intermediate = torch.nn.functional.silu(gate) * up
        
        rss_peak = get_current_rss_mb()
        activation_delta_mb = max(0.0, rss_peak - rss_start)
        
        results.append({
            "context_length": float(ctx_len),
            "hidden_dim": float(hidden_dim),
            "intermediate_dim": float(intermediate_dim),
            "activation_delta_mb": round(activation_delta_mb, 2),
            "peak_rss_mb": round(rss_peak, 2)
        })
        
        del x, qkv, gate, up, intermediate
        
    del w_qkv, w_gate, w_up
    return results


def measure_allocator_drift_without_gc(
    iterations: int = 100,
    layer_size_mb: float = 64.0
) -> Dict[str, float]:
    """Measure memory drift over consecutive token steps without manual gc.collect()."""
    rss_history = []
    num_floats = int((layer_size_mb * 1024 * 1024) / 4)  # FP32 elements
    
    # Pre-allocate one recycled layer buffer to test pure streaming
    recycled_buffer = torch.zeros(num_floats, dtype=torch.float32)
    
    rss_initial = get_current_rss_mb()
    
    for i in range(iterations):
        # Simulate decoding token: batch=1, seq=1, dim=2048
        x = torch.randn(1, 1, 2048, dtype=torch.float32)
        w = recycled_buffer[: 2048 * 2048].view(2048, 2048)
        out = torch.matmul(x, w)
        _ = torch.sum(out)
        
        rss_history.append(get_current_rss_mb())
        del x, w, out
        # Note: Deliberately NO gc.collect() to observe real allocator behavior
        
    rss_final = rss_history[-1]
    peak_rss = max(rss_history)
    drift = rss_final - rss_initial
    
    del recycled_buffer
    
    return {
        "iterations": float(iterations),
        "initial_rss_mb": round(rss_initial, 2),
        "peak_rss_mb": round(peak_rss, 2),
        "final_rss_mb": round(rss_final, 2),
        "drift_mb": round(drift, 2)
    }


def compute_moe_backbone_memory(
    arch_name: str,
    hidden_dim: int,
    num_layers: int,
    vocab_size: int
) -> BackboneMemoryEstimate:
    """Calculate the theoretical memory bound for an MoE model's non-expert shared backbone."""
    # 1. Embedding: vocab_size * hidden_dim
    embedding_params = vocab_size * hidden_dim
    # 2. LM Head (un-tied): vocab_size * hidden_dim
    lm_head_params = vocab_size * hidden_dim
    # 3. Attention projections per layer: Q, K, V, O = 4 * (D * D)
    attention_params = num_layers * (4 * hidden_dim * hidden_dim)
    
    total_backbone_params = embedding_params + lm_head_params + attention_params
    
    # Memory calculations
    mem_fp16_mb = (total_backbone_params * 2.0) / (1024 * 1024)
    mem_int8_mb = (total_backbone_params * 1.0) / (1024 * 1024)
    mem_int4_mb = (total_backbone_params * 0.5) / (1024 * 1024)
    
    # 2 GB budget condition (leaving at least 500 MB for runtime, KV cache, activations)
    fits_in_2gb = mem_int4_mb <= 1500.0
    
    return BackboneMemoryEstimate(
        architecture_name=arch_name,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        vocab_size=vocab_size,
        embedding_params=embedding_params,
        attention_params=attention_params,
        lm_head_params=lm_head_params,
        total_backbone_params=total_backbone_params,
        memory_fp16_mb=round(mem_fp16_mb, 2),
        memory_int8_mb=round(mem_int8_mb, 2),
        memory_int4_mb=round(mem_int4_mb, 2),
        fits_in_2gb_ram=fits_in_2gb
    )


def simulate_moe_expert_disk_reads(
    num_layers: int = 32,
    experts_per_layer: int = 2,
    expert_size_mb: float = 16.0
) -> Dict[str, float]:
    """Measure I/O latency when reading 2 non-contiguous expert chunks per layer on NVMe."""
    temp_dir = tempfile.mkdtemp(prefix="ultron_moe_io_test_")
    
    try:
        # Create a pool of 8 candidate expert files (8 * 16 MB = 128 MB total written)
        total_candidates = 8
        num_floats = int((expert_size_mb * 1024 * 1024) / 4)
        
        for exp_idx in range(total_candidates):
            fpath = os.path.join(temp_dir, f"expert_{exp_idx:02d}.bin")
            data = torch.randn(num_floats, dtype=torch.float32)
            torch.save(data, fpath)
            del data
            
        # Simulate token generation: for each of 32 layers, load top-2 distinct experts sequentially
        read_times_ms = []
        bytes_read = 0.0
        
        t_token_start = time.perf_counter()
        for layer_idx in range(num_layers):
            # Select 2 distinct candidate experts
            chosen = [layer_idx % total_candidates, (layer_idx + 3) % total_candidates]
            for exp_idx in chosen:
                fpath = os.path.join(temp_dir, f"expert_{exp_idx:02d}.bin")
                t0 = time.perf_counter()
                exp_data = torch.load(fpath, map_location="cpu", weights_only=True)
                _ = exp_data[0].item()
                t1 = time.perf_counter()
                read_times_ms.append((t1 - t0) * 1000.0)
                bytes_read += expert_size_mb
                del exp_data
                
        t_token_end = time.perf_counter()
        total_token_sec = t_token_end - t_token_start
        throughput_mb_s = bytes_read / total_token_sec
        tokens_per_sec = 1.0 / total_token_sec
        
        return {
            "num_layers": float(num_layers),
            "experts_per_layer": float(experts_per_layer),
            "expert_size_mb": expert_size_mb,
            "total_bytes_read_mb": round(bytes_read, 2),
            "total_token_io_sec": round(total_token_sec, 3),
            "throughput_mb_s": round(throughput_mb_s, 2),
            "theoretical_max_tokens_per_sec": round(tokens_per_sec, 2),
            "avg_expert_read_ms": round(sum(read_times_ms) / len(read_times_ms), 2)
        }
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def run_audit_experiment() -> Dict:
    """Execute EXP-002 comprehensive audit and produce structured output."""
    # 1. Prefill activation peaks
    activation_peaks = measure_prefill_activation_peaks()
    
    # 2. Allocator drift without GC sweeps
    drift_metrics = measure_allocator_drift_without_gc(iterations=100)
    
    # 3. MoE Backbone Lower Bounds
    llama_70b = compute_moe_backbone_memory("Llama-3-70B Backbone", 8192, 80, 128256)
    mixtral_8x22b = compute_moe_backbone_memory("Mixtral-8x22B Backbone", 6144, 56, 32000)
    compact_moe = compute_moe_backbone_memory("Compact-64x1.5B Backbone", 2048, 28, 32000)
    
    backbone_comparisons = [
        asdict(llama_70b),
        asdict(mixtral_8x22b),
        asdict(compact_moe)
    ]
    
    # 4. MoE Non-Contiguous Expert Disk Read Throughput
    expert_io_metrics = simulate_moe_expert_disk_reads(num_layers=32, experts_per_layer=2, expert_size_mb=16.0)
    
    return {
        "experiment_id": "EXP-002",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prefill_activation_peaks": activation_peaks,
        "allocator_drift_no_gc": drift_metrics,
        "moe_backbone_memory_analysis": backbone_comparisons,
        "moe_expert_io_metrics": expert_io_metrics
    }


if __name__ == "__main__":
    report = run_audit_experiment()
    output_path = os.path.join(os.path.dirname(__file__), "exp002_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"EXP-002 audit results saved to {output_path}")
