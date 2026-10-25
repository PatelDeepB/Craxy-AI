# ULTRON Research Plan: High-Quality AI in 2 GB RAM & 100 GB Storage

## 1. Executive Problem Statement & Physics of the Memory Wall

To execute a 100B parameter dense model in FP16 requires ~200 GB of memory. Even aggressively quantized:
- INT8: ~100 GB
- INT4: ~50 GB
- INT2: ~25 GB
- INT1 (BitNet ternary): ~12.5 GB - 19.7 GB

In all cases, a 100B parameter model cannot reside entirely within a 2,048 MB physical DRAM budget. Furthermore, even small dense models face tight constraints in 2 GB RAM:
- 7B model in 4-bit requires ~3.8 GB (exceeds budget).
- 3B model in 4-bit requires ~1.8 GB weights + 250 MB runtime + 300 MB KV cache = ~2.35 GB (exceeds budget).
- 1B model in 4-bit requires ~600-800 MB weights, leaving headroom for KV cache and runtime, but standard 1B models fall dramatically short of 100B-class performance (e.g. Llama-3.2-1B MMLU is ~49%, compared to ~82% for 70B-100B models).

Therefore, achieving 100B-class capability in 2 GB RAM while utilizing up to 100 GB storage requires breaking the paradigm of static monolithic in-RAM model residency.

---

## 2. Comprehensive Literature Map & Prior Art

We survey peer-reviewed literature, preprints, and open-source implementations addressing memory-constrained inference and storage offloading:

### 2.1 Storage-to-DRAM Streaming & Dynamic Paging
1. **"LLM in a Flash: Efficient Large Language Model Inference with Limited Memory"**
   - *Authors:* Keivan Alizadeh, Iman Mirzadeh, Dmitry Belenko, Karen Khatamifard, Minsik Cho, Carlo C Del Mundo, Mohammad Rastegari, Mehrdad Farajtabar (Apple).
   - *Date:* Dec 20, 2023. *Citation:* arXiv:2312.11514.
   - *Key Finding:* Stores weights in flash (NVMe SSD) and streams only needed parameters into DRAM during generation. Exploits activation sparsity in Feed-Forward Networks (FFNs) (often >90% zeros in ReLU/ReGLU) to load only predicted non-zero neuron weights. Employs "sliding window" activation buffers to reuse memory. Achieved 4-5x speedup on CPU over naive streaming, running models 2x larger than DRAM.
   - *Relevance to ULTRON:* Proves flash-streaming feasibility, but pure FFN sparsity alone cannot bridge a 50x ratio (100B on 100GB disk into 2GB RAM) without additional techniques.

2. **"Fast Inference of Mixture-of-Experts Language Models with Offloading"**
   - *Authors:* Denis Eliseev, Denis Mazur (Yandex / Higher School of Economics).
   - *Date:* Dec 27, 2023. *Citation:* arXiv:2312.17238.
   - *Key Finding:* In MoE architectures (e.g., Mixtral 8x7B), only a subset of experts (top-2 of 8) are active per token. By offloading inactive experts to disk/RAM and pipelining expert loading with attention computation, inference memory drops to the size of the attention backbone plus 2 active experts.
   - *Relevance to ULTRON:* If a 64-expert MoE is stored on a 100 GB SSD, only active experts (~20-50 MB per layer) need to be loaded into the 2 GB RAM budget at any instant.

3. **"EdgeMoE: Fast On-Device Inference of MoE Models via Dynamic Expert Pruning and Paging"**
   - *Authors:* Tse-Chun Zhang, et al.
   - *Date:* Aug 2022. *Citation:* arXiv:2208.13041 / ACM ASPLOS.
   - *Key Finding:* Demonstrated dynamic expert paging from storage to memory on resource-constrained edge devices with non-volatile memory.

### 2.2 Non-Parametric Memory & External Knowledge Retrieval
4. **"Improving Language Models by Retrieving from Trillions of Tokens" (RETRO)**
   - *Authors:* Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford, Katie Millican, et al. (DeepMind).
   - *Date:* Feb 2022. *Citation:* ICML 2022 / arXiv:2112.04426.
   - *Key Finding:* A 7B parameter RETRO model, augmented with a frozen 2-trillion-token retrieval database, matches the factual accuracy and perplexity of a 175B parameter dense model (Jurassic-1 / GPT-3) across diverse language benchmarks.
   - *Relevance to ULTRON:* Proves that 100B-class factual performance does not require 100B parameters in RAM if world knowledge is stored in indexed external storage (fitting comfortably within the 100 GB disk budget).

5. **"DiskANN: Fast Accurate Billion-Point Nearest Neighbor Search on a Single Node"**
   - *Authors:* Harsha Vardhan Simhadri, Ravishankar Krishnaswamy, Gopal Srinivasa, et al. (Microsoft Research).
   - *Date:* NeurIPS 2019.
   - *Key Finding:* Serves billion-scale vector similarity search directly from SSD in under 5 milliseconds with <100 MB of RAM footprint.
   - *Relevance to ULTRON:* Shows that an on-disk knowledge/memory store can be queried at sub-token latency without blowing the 2 GB RAM budget.

### 2.3 Extreme Quantization & Ternary Representations
6. **"The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits" (BitNet b1.58)**
   - *Authors:* Shuming Wang, Hongyu Wang, Shaohan Huang, Furu Wei, et al. (Microsoft Research).
   - *Date:* Feb 27, 2024. *Citation:* arXiv:2402.17764.
   - *Key Finding:* Constrains weights to ternary values {-1, 0, +1}. Replaces floating-point matrix multiplications (GEMM) with pure integer addition, drastically reducing energy, memory bandwidth, and storage. At 3B scale, matches FP16 quality. A 7B model requires only ~1.4 GB for weights.
   - *Relevance to ULTRON:* Ternary weights fit 7B-10B models into <1.5 GB RAM, but alone do not reach 100B-level reasoning without external assistance or scaling.

7. **"QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Vector Quantization"**
   - *Authors:* Albert Tseng, Jerry Chee, Qinghao Hu, et al. (Cornell / Meta).
   - *Date:* Feb 2024. *Citation:* arXiv:2402.04396.
   - *Key Finding:* Achieves state-of-the-art 2-bit post-training quantization with minimal perplexity degradation.

---

## 3. Architectural Candidate Matrix

We identify five core technical pathways that leverage 100 GB storage while respecting 2 GB RAM:

| Approach | Architecture Description | RAM Footprint | Storage Footprint | Primary Bottleneck | Quality Potential |
|---|---|---|---|---|---|
| **A: Sequential Layer Streaming** | Full 70B/100B dense model on NVMe (4-bit, ~40-60 GB). Stream one layer at a time into a 500 MB RAM buffer. | 1.0 - 1.5 GB | 50 - 70 GB | Disk I/O bandwidth (PCIe bus) | Exact 100B parity, but low token/s |
| **B: Sparse MoE with Expert Paging** | 80B-100B MoE model (e.g. 64 experts, top-2 active). Attention backbone in RAM, active experts streamed from SSD. | 1.2 - 1.8 GB | 45 - 80 GB | Expert prefetching & routing latency | High (approaching 100B MoE) |
| **C: Non-Parametric Memory Engine** | Compact 1.5B-3B core reasoner in RAM (~1.0 GB) + 80 GB on-disk knowledge index (DiskANN / dense vectors). | 1.3 - 1.7 GB | 60 - 90 GB | Retrieval quality & reasoning gap of 1.5B | High on knowledge/MMLU, moderate on complex logic |
| **D: Ultra-Quantized Ternary + Speculative Cascade** | BitNet b1.58 (7B ternary in 1.4 GB RAM) as fast draft generator, verified against on-disk streamed MoE layers. | 1.5 - 1.9 GB | 50 - 90 GB | Verification throughput & KV cache pressure | High |
| **E: Naive In-Memory Small Model** | Standard 1B-3B model loaded directly into RAM. | 0.8 - 1.8 GB | 1 - 3 GB | Fails charter: cannot match 100B quality | Fails Charter Target |

---

## 4. Staged Research Roadmap

### Stage 0: Ground Truth Memory Budget & Streaming Feasibility (Turn 1 - Active)
- **Objective:** Establish empirical memory accounting model and test NVMe sequential layer streaming mechanics on the local environment.
- **Hypothesis:** A single transformer layer buffer can be dynamically populated and recycled in memory with zero memory leak, proving that $M_{\text{weights}}$ can be bounded to a single layer size $M_{\text{layer}}$ rather than total model size $M_{\text{total}}$.
- **Deliverables:**
  * Memory benchmark script measuring Python/PyTorch runtime RSS, KV cache scaling, and layer streaming I/O latency.
  * Formal Research Charter and Staged Plan.

### Stage 1: Memory Wall & I/O Bandwidth Bounds (Turns 2-3)
- **Objective:** Mathematically and experimentally bound the maximum possible tokens/second for layer streaming vs expert paging given PCIe NVMe read throughput.
- **Key Question:** For a 70B model in 4-bit (~35 GB total, ~450 MB per layer across 80 layers), how fast can layers be streamed on modern SSDs (3,000 - 7,000 MB/s)?
  * Analysis: 35 GB / 5,000 MB/s = 7 seconds per token (0.14 tokens/s) for full sequential streaming.
  * Conclusion: Pure dense layer streaming is too slow for interactive use without activation reuse or MoE sparsity!
- **Deliverables:** Analytical latency model comparing Dense Streaming vs Sparse MoE Paging vs Retrieval-Augmented Generation.

### Stage 2: MoE Expert Swapping & Flash Activation Sparsity (Turns 4-6)
- **Objective:** Prototype selective expert streaming. In an MoE model, only 2 experts out of 32 or 64 are read per layer.
  * Analysis: 2 active experts = ~15-25 MB per layer. Total I/O per token = 80 layers * 20 MB = 1.6 GB. At 5,000 MB/s, latency = ~320 ms per token (~3 tokens/second).
- **Deliverables:** Prototype of an on-disk MoE expert loader with LRU cache in RAM.

### Stage 3: Retrieval-Augmented Non-Parametric Engine (Turns 7-9)
- **Objective:** Evaluate combining a 1.5B-3B compact reasoning model with an on-disk 50-80 GB dense vector index (DiskANN / FAISS on-disk).
- **Deliverables:** Accuracy comparison on MMLU / GSM8K with vs without on-disk retrieval augmentation.

### Stage 4: Integrated Prototype Synthesis (Turns 10-12)
- **Objective:** Integrate the winning architectural paradigm (e.g. Hybrid MoE + On-Disk Knowledge Index) into a unified inference runtime.
- **Deliverables:** End-to-end inference pipeline executing under 2,048 MB RSS.

### Stage 5: Formal Charter Verification & Red-Team Audit (Turns 13-14)
- **Objective:** Comprehensive evaluation against Tier 1, Tier 2, and Tier 3 acceptance criteria across MMLU, GSM8K, HumanEval, and ARC-Challenge.

---

## 5. Go / No-Go Decision Criteria

1. **Gate 0 (Memory Feasibility):** If runtime RSS + KV cache (2,048 tokens) exceeds 1,500 MB before any weights are loaded, standard PyTorch is deemed unviable and we must pivot to custom C++ / llama.cpp / lightweight ONNX runtime.
2. **Gate 1 (I/O Throughput):** If layer/expert streaming cannot achieve >= 1.0 token/second on reference NVMe SSD, pure weight streaming is rejected in favor of Non-Parametric Retrieval or Speculative Pipelining.
3. **Gate 2 (Quality Floor):** If compact model + retrieval fails to beat standard 7B dense models on MMLU (>65%), retrieval-alone is rejected in favor of Sparse MoE paging.
