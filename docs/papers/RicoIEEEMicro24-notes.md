# AMD XDNA NPU & IRON Programming Interface — Research Notes

*Consolidated from Rico et al. (IEEE Micro, 2024), Hunhoff et al. (FCCM, 2025), and follow-on literature through April 2026. Organized around three threads: (1) the NPU architecture, (2) IRON as a close-to-metal programming interface and its analogy to CUDA, (3) data movement — the Placer system, taplib, and canonical patterns.*

---

## 1. Why IRON / MLIR-AIE exist

AMD's XDNA NPU is a **spatial dataflow** accelerator with software-managed on-chip memory. AMD's mainstream **Ryzen AI Software / Vitis AI EP** flow (PyTorch/TF → ONNX → Vitis AI EP) hides software managed memory behind an ML-oriented abstraction.

**IRON** (*Interface Representation for hands-ON* programming) serves performance engineers, researchers, DSP programmers, and anyone building kernels that don't exist in a library. It sits on top of the **`mlir-aie`** MLIR dialect and gives full control while remaining in Python. Philosophically it is to the NPU what CUDA C + PTX are to an NVIDIA GPU — except it targets a spatial VLIW array rather than SIMT cores.

Key positioning:
- It is **not** an end-to-end compiler for ML models; AMD calls it "complementary, not a replacement" for Ryzen AI Software.
- It targets **researchers, tool-builders, and performance engineers** who want to drive the NPU directly or who are prototyping higher-level compilers that lower to it.
- The codebase split in 2025: `Xilinx/mlir-aie` (the MLIR toolchain) and `amd/IRON` (close-to-metal Python API + applications, including a Llama 3.2 1B demo).

---

## 2. XDNA NPU architecture — the details that matter for programming

### 2.1 The spatial grid

A 2-D grid of tiles connected by a streaming network-on-chip. Tiles are physically arranged in space; position determines connectivity and role.

| Tile type | Count (XDNA 1 / XDNA 2) | Contains | Role |
|---|---|---|---|
| **Compute (AIE) tile** | 20 / 32 | VLIW+SIMD core, 64 KB L1 DMEM, program memory, 2 MM2S + 2 S2MM DMA channels, 16–64 locks, stream switch | Number crunching |
| **Memory tile** | 5 / ~8 | 512 KB L2 SRAM, DMAs, locks, stream switch | L2 scratchpad, broadcast staging, reshape |
| **Shim (Interface) tile** | 4 / ~8 | DMAs with PASID-tagged external access, stream switch | LPDDR5x via SoC fabric. Can be reconfigured to optimize memory access patterns|

On XDNA 1 Phoenix: 4×5 layout — 4 rows of compute, 1 row of memory, 1 row of shim; only four of five columns have shims.

### 2.2 Three levels of explicit memory

| Level | Location | Size (XDNA 1) | Management |
|---|---|---|---|
| **L1** | DMEM per compute tile | 64 KB × 20 = 1.28 MB | Software |
| **L2** | SRAM per memory tile | 512 KB × 5 = 2.5 MB | Software |
| **L3** | External LPDDR5x | up to 128-bit @ 120 GB/s | Driver / host |

This is all software managed. The hardware does not coherence.

A compute tile can directly load/store the L1 DMEM of its **four nearest neighbors** (IRON's "SharedMem" feature). Neighbors can also exchange values via 32-bit streams or **512-bit cascade streams** (a dedicated accumulator-forwarding path, critical for reduction chains).

### 2.3 The VLIW core

Variable-length VLIW (16–128 bit instructions) issuing up to **6 operations in parallel**:

1. **Matrix unit** — one matrix op/cycle, reconfigurable per-cycle for dtype:

| Operands | MACs/cycle |
|---|---|
| INT8 × INT4 | **512** |
| INT8 × INT8 | 256 |
| INT16 × INT16 | 64 |
| BF16 × BF16 | 128 (FP32 accumulation) |

   Supports 50% structured sparsity (doubles effective throughput). XDNA 2 adds **Block FP16** — shared-exponent BFP delivering INT8-class throughput with FP16-class accuracy, no retraining.

2. **Vector add/compare/min-max unit** — parallel to matrix.
3. **Vector shuffle unit** — rearranges elements to the matrix unit's expected layout.
4. **Two load slots** — 256 bits each with 2D/3D AGUs for tensor strides. One slot supports on-the-fly weight decompression.
5. **One store slot** — 256 bits with AGU.
6. **Scalar + control-flow slot**.

Peak performance requires all six slots busy every cycle. This is what Peano-compiled C++ kernels (using AIE API intrinsics) do, and what `core_fn` in IRON wraps.

### 2.4 Data movement hardware

Every transfer is a triple:

```
[ Main Memory to SharedMem DMA on source ] → [ stream switch route ] → [ SharedMem to MainMemory DMA on dest ]
```

Each DMA channel is a **fully independent engine** with its own memory port, stream port, and lock interface. Programmed with a **Buffer Descriptor (BD)** supporting:

- **Up to 5-D addressing** — sizes, strides, offset per dim (this is what `TensorAccessPattern` models directly).
- **Per-dim stride** in elements, independent of size — a single BD can tile a tensor without a surrounding loop.
- **Zero-value padding injection** — halo regions for NxN conv materialized "for free" during the transfer.
- **On-the-fly compression/decompression** — zero-byte bitmask, up to 8× on sparse tensors.
- **BD chaining** — one BD triggers the next, no host intervention.

Synchronization via **hardware locks** (16–64/tile, 64 states each). Switch-level back-pressure is credit-based; tile-level producer/consumer sync is lock-based. Broadcast and multicast are **hardware features of the switch** — one MM2S read → N S2MM sinks in a single pass.

### 2.5 Partitioning and isolation

The array can be sliced **column-wise** into up to 4 disjoint partitions, each PASID-tagged, isolated by column boundary registers. A **command processor** (microcontroller running ERT/MERT firmware) handles partition setup, DMA orchestration, time-sharing, and error management. Most control flow happens at compile time; there is no dynamic instruction scheduling.

### 2.6 XDNA 2 (Strix Point / Strix Halo)

32 AI tiles (up from 20), ~1.6× on-chip memory, 2× MACs/tile. **50 TOPS** INT8 — and **50 TOPS Block FP16**, the first NPU to hit parity. Up to 8 concurrent models, per-column power gating. Same spatial+temporal+hybrid partitioning scheme.

>[note]
> I believe that the NPU Interface tiles can be reprogrammed similar to how an
> FPGA is. I keep seeing references to the NPU 'configuration' or the 'Memory
> tile configuration' but I cannot determine exactly how to do so.
>
> - aa

---

## 3. IRON ↔ CUDA analogy

- **`Worker` ↔ CUDA thread-block** (loosely). Construction `Worker(core_fn, fn_args)` is deliberately patterned on thread libraries, but a Worker is **bound to a physical tile** and runs an independent instruction stream — no oversubscription, no warp scheduler.
- **`ObjectFifo` ↔ shared memory + barriers**. A lock-mediated circular buffer with `acquire(n)` / `release(n)`. Default depth 2 = ping-pong double buffering. On a GPU you'd hand-code this with two shared-memory buffers and `__syncthreads()` between a producer warp and a consumer warp.
- **L1 DMEM ↔ shared memory**. Same role, same size class, same staging pattern.
- **Cascade streams ↔ warp-shuffle reductions**. Both are hw-accelerated neighbor-to-neighbor paths avoiding memory traffic.
- **aiecc ↔ nvcc; Peano ↔ ptxas; XRT ↔ CUDA Driver**.

>[note]
> The AMD Peano compiler is interesting and provides extensions to LLVM IR. It
> is a neat read on its own.
>
> - aa

### 3.1 Where the analogy breaks (important)

- **No SIMT.** 1–32 Workers, not tens of thousands of threads. Latency hidden by double-buffered DMA, not by swapping warps. "Occupancy" isn't a concept; the analog is "are your six VLIW slots busy and are DMAs saturating the switch."
- **Placement is a spatial problem, not an allocation problem.** You decide which shim, which memory tile, which compute tiles, and which route. This is why `Placer` exists at all — GPUs have no analog because their memory routing is fixed.
- **Data movement is a program, not a syscall.** `rt.fill` *looks* like a memcpy but emits BD configurations executed asynchronously against a compile-time switch network.
- **Tensors live across layers.** NPU L2 contents can persist across entire networks. CUDA/OpenCL/SYCL force spills to LLC at kernel boundaries; NPU designs explicitly decide which activations stay in L2 across layer boundaries.
- **Broadcast is a hardware primitive.** One DDR read → N consumer L1s via the switch, not N fetches into N SM caches.
- **Two-language by design.** Python for structure, C++ for the per-core kernel. The split is the *default*. (not an optimization (unlike CUDA + CUTLASS)).
- **No "kernel launch."** Workers typically contain `for _ in range_(sys.maxsize):` — they run as a standing dataflow pipeline fed by the Runtime from the host. Closer to a persistent-kernel CUDA design than a normal CUDA kernel.
- **The higher tier isn't part of IRON.** No cuBLAS/cuDNN equivalent *inside* IRON. That role is played by separate projects stacked on top (Ryzen AI Software, MLIR-AIR, Triton-XDNA, ARIES).

### 3.2 One-sentence summary

**IRON is to the XDNA NPU what CUDA C + PTX + CUDA Driver are to an NVIDIA GPU — a close-to-metal, vendor-blessed programming interface exposing the hardware's defining features — except the NPU's defining feature is explicit spatial dataflow, so where CUDA gives you threads and implicit data movement, IRON gives you Workers, ObjectFifos, and a Placer.**

> (Lol, wonder where claude found the term 'vendor-blessed' - aa)

On a GPU you describe *what to compute* and the hardware figures out *where and when*. On the NPU you describe *what to compute, where it runs, how data gets there, and when it's synchronized* — and the payoff is deterministic latency and 4–33× better perf/W on workloads the architecture is shaped for.

---

## 4. Data movement — the part that most shapes performance

> Note that to fully grasp this you need to read the section on memory
> management in the XDNA NPU Architecture paper.
>
> - aa

### 4.1 The abstraction stack over the DMA hardware

Three levels:

- **Level 0 — Raw DMA programming.** `shim_dma_single_bd_task(...)` with hand-filled sizes/strides. Still supported via `inline_ops`. Necessary for unusual patterns.
- **Level 1 — `ObjectFifo`.** The primary abstraction. A named logical channel between producer and consumer endpoints, backed by a circular buffer of configurable depth (default 2 for ping-pong). The compiler picks concrete buffers, DMA BDs, switch routes, and locks. `prod()`/`cons()` handles obtain endpoint-specific views; multiple `cons()` calls create a broadcast. `acquire(n)`/`release(n)` give **sliding-window** semantics (hold context, advance by one).
- **Level 2 — `ObjectFifoHandle.forward/split/join`.** L2 staging patterns, described below.

### 4.2 `taplib` — reasoning about BD access patterns

Exists because 5-D sizes+strides programming is error-prone, hardware-constrained, and the single most common source of bugs in IRON designs (source??? Probably the Halstead-effort criterion).

**`TensorAccessPattern` (tap)** — one DMA transfer over a logical tensor:
```python
tap = TensorAccessPattern(
    tensor_dims=(6, 4),       # logical tensor shape
    offset=0,                 # starting element offset
    sizes=[1, 1, 3, 2],       # up to 5-D sizes
    strides=[0, 0, 4, 1],     # per-dim strides (in elements)
)
```
The sizes/strides tuple is literally what the BD register is programmed with.

**`TensorAccessSequence` (tas)** — a list of taps sharing a tensor shape; the object handed to `rt.fill(...)` in place of raw sizes/strides.

**Access maps — the debugging superpower:**
- `access_count()` — per-element visit count (array over the tensor).
- `access_order()` — per-element visit number (0-indexed).

These turn "is this DMA pattern correct?" into a unit-testable question:
```python
assert taps0.access_count().max() == 1   # no element touched twice
assert tap00.access_order().max() == 5   # 6 elements visited, 0..5
```
The heat maps in Figure 3 of the paper are these arrays visualized.

**Access-equivalence.** Two taps are access-equivalent if their count+order maps match, even if their (dim, size, stride, offset) tuples differ. This matters because different DMA units have different max dimensionality, per-dim size limits in bits, and switch-configuration constraints — so a logically-correct tap may need to be rewritten to a legal form for the specific channel it will run on, without changing semantics.

**`TensorTiler2D`** — a 277-SLOC tap generator covering: tile size, row/column access within a tile, tile grouping, row/column access of groups, group repeats, group steps. Used by 5 of 27 example designs (MSAdd, MVAdd, MVMul, GEMM, MTranspose); credited with most of the GEMM Halstead-effort reduction. It's the extensibility point for custom tiling patterns — write a new generator, never touch sizes/strides by hand again.

### 4.3 The Placer system

Placement is the *spatial* half of the problem that access patterns are the *temporal* half of — and they're coupled, because routing depends on producer/consumer locations.

**What placement assigns:**
- Compute tiles — which (col, row) for each Worker.
- Memory tiles — which location for each L2-staged ObjectFifo.
- Shim tiles — which shim for each DDR-endpoint ObjectFifo.

**Constraints:**
- Fixed per-column topology (not every column has a shim on XDNA 1).
- L1-neighbor access limited to 4 nearest tiles.
- Finite DMA channels and locks per tile.
- Stream-switch configuration limits.
- Cascade streams follow fixed geometric patterns — reduction chains need contiguous layouts.

**The interface.** `Placer.make_placement()` is called during `resolve_program`; it invokes `.place(tile)` on every `Placeable` before MLIR generation. Partial placement is supported; hints `AnyShim`/`AnyMemTile`/`AnyComputeTile` constrain by tile *kind* without picking a specific one.

**The reference `SequentialPlacer`** (64 SLOC): assigns Workers in grid order, keeps memory-tile and shim-tile endpoints in the same column as their compute endpoints. Covers 24/27 example designs. Falls through on:
- **BBlock** — mostly auto, some hand-placement for pipeline stages.
- **ResNetConv2x** — fully hand-placed; too much resource pressure.
- **GEMM** — fully hand-placed; specific cascade-stream geometry.

These failures are the whole point of exposing the interface — placement is an active research area, and the `Placer` API is the hook for simulated annealing, ILP solvers, ML-guided search, or external DSE tools (wrappable via standard language-binding techniques into a Python `Placer`).

### 4.4 Canonical data-movement patterns

The recurring shapes across the example designs. These are the vocabulary.

| Pattern | Shape | When to use |
|---|---|---|
| **L2 buffering** (`forward`) | shim → memory tile → compute tile(s) | Weights reused many times; pay DDR cost once |
| **Broadcast** | 1 MM2S → N S2MM via switch fanout | Same tile to many cores (e.g., GEMM row/col broadcast) |
| **Split / join** | L2 tile sub-distributed to cores / partial outputs aggregated in L2 | Conserves shim channels and DDR bandwidth |
| **Sliding window** | ObjectFifo depth ≥ window size, `acquire(1)`+`release(1)` to advance | Stencils, convolution, FIR filters |
| **Pipelining** | One Worker per stage, ObjectFifos between; larger depth on variable-latency stages | Multi-stage ops (bottleneck block, edge detect) |
| **Data-parallel tiling** | N identical Workers, `TensorTiler2D`-generated tas distributes tiles | "SIMT-equivalent" — same code, different data, but explicit |
| **Reduction via cascade** | Adjacent cores, matrix-unit output → next core's cascade input | Sum-reduce across N cores without memory round-trip |
| **Skip connections** | Long-lived larger-depth ObjectFifo; explicit placement to avoid route conflicts | Residual networks |

### 4.5 Access patterns the matrix unit actually wants

Once data is in L1, separate requirements apply for peak throughput:

- **Innermost dimension contiguous** — to feed a full 256-bit vector load per cycle. Often requires rewriting a tap from its logical form.
- **Inner tile shape matches matrix-unit shape.** Good GEMM tiles at 3 levels: `(M,K,N) → (m,k,n) → (r,s,t)` where `(r,s,t)` matches matrix-unit native shape, middle fits in L1, outer fits in L2.
- **Ping-pong depth-2 is mandatory** for hiding DMA latency — ObjectFifo default handles it, but don't break it (no depth-1, don't hold acquires too long).
- **Pre-shuffle via DMA** when possible — the vector shuffle slot exists, but shuffling during transfer (via sizes/strides) reduces shuffle pressure at compute time. The NPU equivalent of "coalesced memory access" tuning.
- **Halo via DMA padding** — zero-injection during transfer keeps the inner kernel loop branch-free and the VLIW slots productive.

### 4.6 The three orthogonal axes

A full NPU design is a specification across three axes that are *separately addressable*:

| Axis | Described by | Lowered to |
|---|---|---|
| **Structure** — what computes what | Workers, ObjectFifos, core_fns | AIE cores, switch config, Peano-compiled kernels |
| **Placement** — where things run | Placer + `.place()` | Specific (col, row) per tile; switch routes |
| **Access** — which bytes move in what order | taps, tases, ObjectFifo depths | DMA buffer descriptors; lock sequences |

You can write a structurally correct design, hand it to `SequentialPlacer`, use default depths, and have something that runs. Then swap in a hand-tuned Placer, a custom tap generator, or adjust depths on critical-path ObjectFifos — **each axis independently**. This separability is why the paper leans so hard on "extensibility," and why external projects like MLIR-AIR can generate the structure axis while leaving placement and access to IRON's own tooling.

---

## 5. The broader ecosystem built on these two papers

The papers have become the anchor references for a growing open-source NPU compilation stack.

- **MLIR-AIR** (Wang et al., arXiv 2510.14871, Oct 2025) — AIR dialect above `mlir-aie`, bridging high-level control flow (Linalg, SCF, Triton, Torch-MLIR, IREE, TOSA) to spatial hardware. `air.launch`/`air.segment`/`air.herd` scoping + `air.channel.put/get` + `air.token` SSA values encoding RAW/WAR/WAW dependencies. Lowers to `mlir-aie` for AMD NPUs, LLVM for other targets. 78.7% compute efficiency on GEMM (matching hand-tuned MLIR-AIE); LLaMA-2 MHA in ~150 lines. **The "high-level compiler tier" the IRON paper gestured at.**
- **ARIES** (Zhuang et al., FPGA '25) — Unified MLIR flow for AIE-core + AIE-graph + FPGA PL. 4.92 TFLOPS FP32 / 15.86 TOPS INT16 / 45.94 TOPS INT8 on Versal GEMM; 22.58× over Riallto on ResNet residual on Ryzen AI NPU.
- **Triton-XDNA** (AMD, 2025) — End-to-end Triton → triton-shared (Linalg) → Transform dialect → MLIR-AIR → MLIR-AIE → XRT binary. Performance parity with hand-written kernels for dense I8/I16/BF16 matmul.
- **Bare-metal GPT-2 fine-tuning** (Rösti et al., arXiv 2504.03083, April 2025) — First client-side LLM *training* on AMD NPU via IRON. 2.8× GEMM speedup, 1.7× end-to-end FLOPS/s on mains, 1.4× energy efficiency. Independent third-party stress-test of IRON's expressivity.
- **GEMM across XDNA generations** (arXiv 2512.13282, Dec 2025) — Unified GEMM framework for XDNA/XDNA 2, among the first published benchmarks leveraging XDNA 2's bfp16.
- **NPUEval** (Kalade et al., arXiv 2507.14403, July 2025) — 102-kernel benchmark for LLM-based NPU kernel generation, built on MLIR-AIE + IRON. SOTA LLMs average ~10% vectorization score even with compiler feedback.
- **AMD tutorials** at ISCA'25, IPDPS'25, MICRO'24, ASPLOS'24.

---

## 6. Future directions

**Short-term (visible today):**
- Three-tier open-source stack stabilizing: Triton/Torch-MLIR/TOSA → **MLIR-AIR** → **MLIR-AIE / IRON** → Peano (LLVM-AIE).
- Block FP16 adoption across SD 3 Medium (Amuse 3.1) and academic GEMM work.
- Kernel library maturation in `amd/IRON`; NPUEval gives a benchmark target.

**Medium-term (roadmap signals):**
- **Ryzen AI 400 "Gorgon Point" (2026)** — Zen 5 + RDNA 3.5 + XDNA 2 refresh; some SKUs targeting ~60 NPU TOPS.
- **"Medusa Point" (2027)** — Zen 6 + RDNA 5 + **next-gen XDNA**. AMD publicly claims 10× client AI perf trajectory.
- Linux kernel-side consolidation via `amdxdna` driver (spatial partitioning, PASID isolation, multi-context scheduling already upstream).

**Longer-term / speculative:**
- **Chiplet-class XDNA.** Several analysts and AMD's own "tile architecture" framing suggest the natural trajectory is promoting XDNA to an Infinity-Fabric-attached chiplet dropping into EPYC/Ryzen/edge SoCs. Early signs: ROCm acknowledgement of XDNA, `amdxdna` upstream, device-generic IRON tooling (Phoenix, Strix, Versal from one codebase).
- **Design-space-exploration as first-class IRON citizen** — the `Placer` interface is the obvious research hook; MLIR-AIR's herd-mapping experiments (1×4 vs 2×2 vs 4×1) are early examples.
- **Visualization & debugging.** `taplib`'s access-count/access-order heatmaps point toward a DaCe-style interactive data-movement debugger for spatial accelerators.

---

## References

1. **Xilinx/mlir-aie** — MLIR toolchain. https://github.com/Xilinx/mlir-aie
2. **amd/IRON** — Close-to-metal programming. https://github.com/amd/IRON
3. Rico, A. et al. "AMD XDNA NPU in Ryzen AI Processors." *IEEE Micro* 44(6), Nov/Dec 2024. DOI: 10.1109/MM.2024.3423692
4. Hunhoff, E. et al. "Efficiency, Expressivity, and Extensibility in a Close-to-Metal NPU Programming Interface." *FCCM 2025*. DOI: 10.1109/FCCM62733.2025.00043 (arXiv:2504.18430)
5. Wang, E. et al. "From Loop Nests to Silicon: Mapping AI Workloads onto AMD NPUs with MLIR-AIR." arXiv:2510.14871, Oct 2025.
6. Zhuang, J. et al. "ARIES: An Agile MLIR-Based Compilation Flow for Reconfigurable Devices with AI Engines." *FPGA '25*. DOI: 10.1145/3706628.3708870
7. **amd/Triton-XDNA** — https://github.com/amd/Triton-XDNA
8. Rösti, A. et al. "Unlocking the AMD Neural Processing Unit for ML Training on the Client Using Bare-Metal-Programming Tools." arXiv:2504.03083, April 2025.
9. "Striking the Balance: GEMM Performance Optimization Across Generations of Ryzen AI NPUs." arXiv:2512.13282, Dec 2025.
10. Kalade, S. et al. "NPUEval: Optimizing NPU Kernels with LLMs and Open Source Compilers." arXiv:2507.14403, July 2025.
11. AMD. "Leveraging the IRON AI Engine API to Program the Ryzen AI NPU" — IPDPS'25 & ISCA'25 tutorials.
12. **amdxdna** Linux kernel driver. https://docs.kernel.org/accel/amdxdna/amdnpu.html
13. AMD XDNA technology page. https://www.amd.com/en/technologies/xdna.html
14. AMD Client AI PC roadmap (Financial Analyst Day 2025) — Gorgon Point (2026), Medusa Point (2027).
