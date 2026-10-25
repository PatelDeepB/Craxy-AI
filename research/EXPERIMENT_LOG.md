# ULTRON Experiment Log

## EXP-001: Empirical Memory Budget Profile & Sequential Layer Streaming Feasibility

- **Date:** 2026-10-25
- **Author:** ULTRON 1
- **Status:** COMPLETED
- **Script:** `experiments/exp001_memory_streaming_benchmark.py`
- **Test Suite:** `tests/test_memory_accounting.py` (4/4 tests passed)
- **Artifact:** `experiments/exp001_results.json`

### 1. Objective and Hypothesis
- **Hypothesis:** By dynamically streaming layers sequentially from local non-volatile storage into a single recycled working buffer in RAM, model weights do not need to reside simultaneously in memory. The process resident set size (RSS) will remain strictly bounded by $M_{\text{runtime}} + M_{\text{layer}} + M_{\text{KV}} \ll 2048\text{ MB}$, with zero cumulative memory drift over 32 layers.
- **Goal:** Establish empirical baseline values for Python/PyTorch runtime RSS overhead, quantify KV cache scaling across context lengths and precisions, and measure SSD layer streaming throughput on the local environment.

### 2. Experimental Environment & Setup
- **OS:** Windows 11 (64-bit)
- **CPU:** 13th Gen Intel(R) Core(TM) i5-13420H (8 physical cores, 12 logical processors)
- **RAM Total:** 16,076 MB (16 GB physical, tested under strict <= 2,048 MB budget enforcement)
- **Storage:** Local NVMe SSD (C: drive with ~198 GB free space)
- **Python:** 3.11.15
- **PyTorch:** 2.13.0+cpu
- **Memory Tracking Tool:** `psutil.Process().memory_info().rss`

### 3. Empirical Results

#### A. Runtime Environment Overhead
| Metric | Measured Value | Notes |
|---|---|---|
| Base Process RSS (Python + PyTorch CPU) | 190.19 MB | Framework baseline floor |
| Available Budget under 2,048 MB Ceiling | 1,857.81 MB | Remaining headroom for weights + KV + activations |

#### B. Key-Value (KV) Cache Memory Scaling
Architecture parameters: $L=32$ layers, $H_{kv}=8$ heads (Grouped-Query Attention), $D=128$ head dim, $B=1$.

| Context Length (tokens) | Precision | Theoretical Size | Layer Tensor Size | Measured RSS Delta |
|---|---|---|---|---|
| 512 | FP16 (2 bytes) | 64.0 MB | 2.0 MB | 2.96 MB |
| 512 | INT8 (1 byte) | 32.0 MB | 1.0 MB | 0.05 MB |
| 1,024 | FP16 (2 bytes) | 128.0 MB | 4.0 MB | 2.00 MB |
| 1,024 | INT8 (1 byte) | 64.0 MB | 2.0 MB | 2.05 MB |
| 2,048 | FP16 (2 bytes) | 256.0 MB | 8.0 MB | 6.00 MB |
| 2,048 | INT8 (1 byte) | 128.0 MB | 4.0 MB | 0.00 MB |
| 4,096 | FP16 (2 bytes) | 512.0 MB | 16.0 MB | 8.00 MB |
| 4,096 | INT8 (1 byte) | 256.0 MB | 8.0 MB | 8.05 MB |
| 8,192 | FP16 (2 bytes) | 1,024.0 MB | 32.0 MB | 24.00 MB |
| 8,192 | INT8 (1 byte) | 512.0 MB | 16.0 MB | 0.00 MB |

*Finding on KV Cache:* At 8,192 context length in FP16, KV cache alone consumes 1,024 MB (50% of the entire 2 GB RAM budget). INT8 or INT4 KV cache quantization is essential for context lengths >= 4,096 tokens.

#### C. Sequential Layer Streaming Benchmark
Simulated 32 transformer layers of 64.0 MB each (total model size on disk = 2,048 MB = 2.0 GB).

| Metric | Measured Value | Threshold / Target | Pass/Fail |
|---|---|---|---|
| Model Size on Disk | 2,048.0 MB | <= 100 GB | PASS |
| Peak Process RSS | 288.39 MB | <= 2,048 MB | PASS (86% headroom) |
| Minimum Process RSS | 288.38 MB | - | - |
| Memory Drift (Layer 32 vs Layer 1) | 0.01 MB | <= 1.0 MB | PASS (No memory leak) |
| Average Layer Load Time | 50.83 ms | - | - |
| NVMe Read Throughput | 1,259.11 MB/s | - | High bandwidth |
| Full 32-layer Pass Time (per token) | 1.627 s | <= 1.0 s target | FAIL (Latency bottleneck) |

### 4. Critical Bottleneck Isolation & Falsification
1. **Memory Feasibility Confirmed:** Sequential layer recycling operates at a peak RSS of only 288.39 MB while executing a 2.0 GB model. Memory leaks are negligible (0.01 MB drift).
2. **Pure Dense Streaming Latency Bottleneck Discovered:** A full pass over 2.0 GB of weights at 1,259 MB/s requires 1.627 seconds per token (~0.61 tokens/second). For a 70B-100B dense model quantized to 4-bit (~35-50 GB), streaming the entire model from disk per token would require 28-40 seconds per token (~0.025 tokens/second).
3. **Implication for Architecture:** Naive sequential dense streaming of a 100B model from disk is refuted as an interactive solution. To achieve target interactive speeds (>= 2 tokens/s), the architecture MUST either:
   - Use **Sparse Mixture-of-Experts (MoE)** where only active experts (~20-50 MB per layer) are loaded from disk per token (reducing I/O by 80-90%), OR
   - Use a **Retrieval-Augmented Core Reasoner** where a 1.5B model stays resident in RAM (~800 MB) and queries the 100 GB storage as non-parametric memory via fast vector search.
