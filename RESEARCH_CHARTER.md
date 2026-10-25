# ULTRON Research Charter

## 1. Mission and Core Objective

The mission of Project ULTRON (Craxy-AI) is to rigorously investigate, design, and validate methods for running a high-quality artificial intelligence model within:
- **Maximum System RAM Budget:** 2,048 MB (2.0 GB) peak resident set size (RSS).
- **Maximum Non-Volatile Storage Budget:** 100 GB (NVMe / SSD flash storage).
- **Target Quality Standard:** Performance parity or superiority relative to modern ~70B-100B+ parameter open-weights dense/sparse foundational models (e.g., Llama-3-70B-Instruct, Qwen-2.5-72B-Instruct, Mixtral-8x22B) on standardized, reproducible evaluation benchmarks.

This is an empirical and architectural research investigation. Feasibility is not assumed a priori. Every claim must be supported by reproducible calculation, published scientific evidence, or direct experimental measurement.

## 2. Hard Constraints and Boundaries

1. **Memory Budget (Strict Ceiling: 2,048 MB RSS):**
   - Measurement: Process Resident Set Size (RSS) as reported by the host operating system (`psutil.Process().memory_info().rss`).
   - The 2,048 MB ceiling must encompass:
     * Model weights resident in RAM.
     * Key-Value (KV) cache for the active context window.
     * Activation buffers and working memory during forward passes.
     * Runtime and framework overhead (Python interpreter, PyTorch / ONNX / C++ runtime, memory allocator fragmentation).
     * Operating system file buffers directly locked or consumed by the inference process.
   - Virtual memory allocations (VMS) backed by swap or lazy pages are not acceptable if accessing them causes memory thrashing that exceeds physical memory capacity.

2. **Storage Budget (Strict Ceiling: 100 GB):**
   - Total disk footprint across all model weights, quantization tables, vector/retrieval indices, and configuration artifacts must not exceed 100.0 GB (100 * 10^9 bytes or 93.13 GiB).

3. **Latency and Usability Requirement:**
   - Inference must produce tokens at an interactive or operational pace (target: >= 2 to 5 tokens/second on standard consumer hardware, with single-query retrieval latency <= 1.0 second).
   - An architecture that fits within 2 GB RAM but requires minutes per token is cataloged as a latency-bound bottleneck, requiring explicit mitigation before passing stage gates.

4. **Safety and Repository Boundaries:**
   - All research must execute locally within this repository.
   - No external API crutches that bypass local execution (e.g., querying external commercial 100B APIs as the actual inference engine) are permitted as the solution. External APIs may only be used as evaluation judges if permitted by human coordinator.

## 3. Operational Definition of "Better than 100B Parameter Model"

Claiming "quality equal to or better than a 100B parameter model" requires concrete, standardized evaluation suites rather than subjective impression. The reference baseline targets are established as:

### 3.1 Reference Baselines
- **Primary Reference Standard:** Llama-3-70B-Instruct and Qwen-2.5-72B-Instruct.
- Reference performance levels (reported 0-shot / 5-shot published scores):
  * **MMLU (Massive Multitask Language Understanding):** ~80.0% to 84.0%
  * **GSM8K (Grade School Math 8K):** ~85.0% to 92.0%
  * **HumanEval (Python Code Generation, pass@1):** ~75.0% to 85.0%
  * **ARC-Challenge (Reasoning):** ~85.0% to 90.0%
  * **IFEval (Instruction Following Strict Prompt):** ~78.0% to 85.0%

### 3.2 Tiered Acceptance Criteria

To make research progress falsifiable and measurable across turns, we define three successive acceptance tiers:

- **Tier 1 (Sub-2GB Structural Baseline):**
  * Peak RAM <= 2,048 MB verified under continuous generation.
  * Context window >= 2,048 tokens without out-of-memory (OOM).
  * Storage <= 100 GB.
  * Task: Zero-shot perplexity and coherent generation verified on standardized test set (e.g. WikiText-2 / C4).

- **Tier 2 (Domain Competence & Knowledge Retrieval):**
  * Peak RAM <= 2,048 MB.
  * Knowledge retrieval latency <= 50 ms per token/chunk from the 100 GB on-disk store.
  * MMLU >= 65.0%, GSM8K >= 60.0% (surpassing standard 3B-7B dense models).

- **Tier 3 (Charter Target Parity):**
  * Peak RAM <= 2,048 MB.
  * Storage <= 100 GB.
  * Average benchmark score across MMLU, GSM8K, and HumanEval within 95% of or exceeding Llama-3-70B-Instruct.
  * Generation throughput >= 2.0 tokens/second on reference commodity CPU.

## 4. Verification and Governance

- Every hypothesis proposed by ULTRON 1 must include:
  1. Theoretical mechanism explaining how the memory wall is circumvented.
  2. Mathematical memory accounting ($M_{\text{weights}} + M_{\text{KV}} + M_{\text{activations}} + M_{\text{runtime}} \le 2048\text{ MB}$).
  3. Falsification criteria (what specific result proves the idea flawed).
- ULTRON 2 must independently verify memory calculations, audit code for memory leaks, verify reproduction, and evaluate failure modes before any approach is marked as validated.
