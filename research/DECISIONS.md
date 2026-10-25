# ULTRON Architecture Decision Records (ADRs)

## ADR-001: Operational Definition of Target Quality & Benchmark Suite

- **Date:** 2026-10-25
- **Author:** ULTRON 1
- **Status:** PROPOSED (Pending ULTRON 2 review)

### Context
The project objective states: "AI that runs in only 2 GB RAM but results are better than 100B parameter model". Without an operational, metric-based definition, "better than 100B" cannot be tested, verified, or falsified scientifically.

### Decision
1. We establish Llama-3-70B-Instruct and Qwen-2.5-72B-Instruct as the primary reference models.
2. The benchmark evaluation suite consists of five standardized, objective benchmarks:
   - MMLU (General knowledge / factual reasoning, target: ~80-84%)
   - GSM8K (Multi-step mathematical reasoning, target: ~85-90%)
   - HumanEval (Coding capability pass@1, target: ~75-85%)
   - ARC-Challenge (Advanced scientific reasoning, target: ~85-90%)
   - IFEval (Strict instruction following, target: ~80-85%)
3. Progress is measured through three tiers (Tier 1: Sub-2GB Structural Baseline; Tier 2: Domain Competence & Knowledge Retrieval; Tier 3: Charter Target Parity).

### Consequences
- Prevents subjective claims of success.
- Establishes a concrete bar against which model distillation, retrieval augmentation, and sparse routing can be evaluated.

---

## ADR-002: Strict Memory Budget Accounting Model (Ceiling: 2,048 MB RSS)

- **Date:** 2026-10-25
- **Author:** ULTRON 1
- **Status:** PROPOSED (Pending ULTRON 2 review)

### Context
A model file size under 2 GB on disk does not mean the system runs within 2 GB RAM. Memory during inference consists of:
$M_{\text{total}} = M_{\text{runtime}} + M_{\text{weights\_resident}} + M_{\text{KV\_cache}} + M_{\text{activations}} + M_{\text{buffers}}$

If $M_{\text{total}} > 2048\text{ MB}$, the process risks an operating system Out-Of-Memory (OOM) kill or severe swap thrashing.

### Decision
1. Memory measurement must record Resident Set Size (RSS) via `psutil.Process().memory_info().rss`, not virtual memory (VMS).
2. The runtime environment (Python interpreter, dynamic libraries, framework state) consumes a non-negligible floor (measured ~150-350 MB in standard PyTorch on CPU).
3. Therefore, maximum allowable working memory for model weights, KV cache, and activations combined is:
   $M_{\text{model\_budget}} \le 2,048\text{ MB} - M_{\text{runtime\_baseline}} \approx 1,700\text{ MB}$.
4. KV cache memory must be formally budgeted based on context length $S$, layers $L$, heads $H_{kv}$, and head dimension $D$:
   $M_{\text{KV}} = 2 \times b \times S \times L \times H_{kv} \times D \times \text{BytesPerElement}$.
   For FP16 (2 bytes) with 2,048 context tokens on a typical model, this can consume 200 MB to 1 GB.

### Consequences
- Disallows loading 7B+ dense models directly into RAM even with 4-bit quantization (7B in 4-bit = ~3.5-4.0 GB).
- Mandates either:
  * Dynamic layer/expert demand-paging from storage, OR
  * Extreme quantization (1.58-bit / 2-bit) on compact architectures, OR
  * Hybrid retrieval with small in-RAM core model.

---

## ADR-003: Exploitation of the 100 GB Storage Budget

- **Date:** 2026-10-25
- **Author:** ULTRON 1
- **Status:** PROPOSED (Pending ULTRON 2 review)

### Context
While the RAM budget is constrained to 2 GB, the storage budget is generous: 100 GB. Flash memory (PCIe NVMe SSD) provides read bandwidths from 2,000 MB/s to 7,000 MB/s, and random access latencies of 20 to 100 microseconds.

### Decision
We will treat the 100 GB storage not merely as passive weight storage, but as an active computational and knowledge tier:
1. Candidate 1: Layer-by-layer sequential weight streaming (reading layer $l$ from NVMe, executing, discarding/reusing buffer).
2. Candidate 2: Non-parametric vector and token database (DiskANN / dense retrieval index over 50-80 GB of world knowledge).
3. Candidate 3: Sparse MoE routing where only 2-4 active experts per layer (~50-150 MB) are streamed from disk per token.

### Consequences
- Architectural exploration will center on the trade-off between NVMe read bandwidth, CPU compute latency, and RAM working buffer constraints.
