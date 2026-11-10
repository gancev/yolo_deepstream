# LOCI x NVIDIA DeepStream — the YOLO CPU post-processor demo

NVIDIA's DeepStream YOLO sample ships a ~200-line C++ output parser,
`nvdsparsebbox_Yolo.cpp`, that runs on the Jetson's CPU after every TensorRT
inference. NVIDIA benchmarks it against its CUDA twin on a Jetson AGX Orin and
publishes the FPS table. This folder cross-compiles that file for a Jetson Thor
(Neoverse V3AE cores, modeled with LOCI's Neoverse V2 target) on a laptop, puts
a LOCI contract around `NvDsInferParseCustomYoloV7`, and stages the scenario:

> An AI agent is asked to "fix the threshold handling and add the missing NMS".
> It fixes the two real latent bugs and adds an O(n^2) IoU loop over the 25,200
> candidate rows. The unit test (a handful of boxes) passes. LOCI flags the new
> loop and the timing multiple from the compiled aarch64 object and fails the
> contract. The acceptable fix (threshold first, NMS left to DeepStream's
> cluster stage) passes.

No Jetson, no GPU and no NGC login are needed; everything is read from the
compiled artifact.

**New here? Read `DEMO.md`**: the story, the contract, LOCI's measurements, the
videos and the upstream fix in one walkthrough.

The whole scenario on one page: `docs/scenario-infographic.html` (download and
open in a browser). Side-by-side videos of the three parser versions are in
`demo/out/`; the LOCI stack map is `docs/loci-stack-map.html`.

## Quick start

```bash
./bootstrap.sh            # clone the pinned upstream, fetch the headers (see third_party/NOTICE)
./build.sh                # aarch64 .o + .so (what LOCI measures), unit test binaries
./test.sh --no-loci       # build + unit test (host, and aarch64 under docker)
# then INSTRUCTIONS.md steps 3-5: loci init, contract accept
./test.sh                 # baseline PASS -> innocent edit FAIL -> fix PASS -> reset
```

Scripts: `scenario/apply.sh innocent|fix`, `scenario/revert.sh`. Prompts for the
live version are in `PROMPTS.md`; the contract and its derivation in
`CONTRACT.md`; what is verified in `STATUS.md`; what only you can supply in
`NEEDS.md`.

## Target and clock

LOCI recipe: target `neoverse-v2`, TU compiled `-O2 -g -mcpu=neoverse-v2
-std=c++11 -fPIC` with GCC 15.2, clock **2601 MHz** — the Jetson Thor T5000's
default 120 W `nvpmodel` CPU clock. Thor's CPU is 14 Arm Neoverse V3AE cores;
LOCI's nearest model is Neoverse V2 (trained at 2.8 GHz, scaled linearly to
2601 MHz). The same recipe at `loci frequency set 1728MHz` approximates a
Jetson AGX Orin (12 Cortex-A78AE at the 30 W default; 2201.6 MHz MAXN) — the
core model stays V2, so take those figures as indicative;
`cmake/aarch64-cortex-a78ae.cmake` is kept for reference and unused.

## Why this file

- **It is NVIDIA's, and it has a published CPU-vs-GPU baseline.** The sample's
  README gives end-to-end FPS for the same pipeline with the CUDA parser and
  with this CPU parser, on a Jetson AGX Orin 64GB (MAXN, GPU 1.3 GHz, 12-core
  CPU 2.2 GHz, 1080p, YOLOv7)
  ([NVIDIA-AI-IOT/yolo_deepstream, Performance](https://github.com/NVIDIA-AI-IOT/yolo_deepstream#performance)):

  | precision | streams | batch | trtexec FPS | deepstream-app, cuda post-process | deepstream-app, cpu post-process |
  |---|---|---|---|---|---|
  | FP16 | 1 | 1 | 126 | 124 | 120 |
  | FP16 | 16 | 16 | 162 | 145 | 135 |
  | INT8 | 1 | 1 | 180 | 175 | 128 |
  | INT8 | 16 | 16 | 264 | 264 | 135 |

  At 120 FPS a frame is 8.3 ms. Once the GPU is fast (INT8) the CPU parser is
  the bottleneck: 1/128 s - 1/175 s attributes roughly 2.1 ms per frame to the
  CPU post-process path (a derived figure, not a published one). NVIDIA has not
  published the same table for Thor; the Orin numbers are the anchor and Thor
  is the deployment the demo prices.
- **It has two real latent bugs for the agent to "fix".** The YOLOv4 path starts
  `maxIndex = -1` and indexes `perClassPreclusterThreshold[maxIndex]` on a row
  whose class scores are all zero (`src/nvdsparsebbox_Yolo.cpp:109,121`); the
  YOLOv7 path never applies the pre-cluster threshold at all and assigns the
  argmax to a `maxIndex` it first filled from `data[5]` (`:249,262`). And the
  README says the post-processor does not include NMS.
- **It builds in minutes.** One translation unit, `-std=c++11 -fPIC`, three
  include directories; no CUDA kernel on the CPU path.

## The parallel with the ArduCopter sprayer demo

| | ArduCopter sprayer (ROSCon 2026) | DeepStream YOLO parser (this folder) |
|---|---|---|
| the patient | `AC_Sprayer::update()` in a 400 Hz scheduler, STM32H7 | `NvDsInferParseCustomYoloV7`, once per inferred frame, Jetson Thor CPU (Neoverse V3AE, modeled as V2) |
| the innocent request | "smooth the ground speed with a median filter" | "fix the threshold handling and add the missing NMS" |
| what the agent writes | a 1024-sample sort on the stack | bug fixes plus a pairwise IoU loop over all 25,200 rows |
| what says go | SITL flies the mission | the handful-of-boxes unit test passes (host and aarch64) |
| what says no | LOCI: 4,240 B against a 512 B stack bound, timing floor | LOCI: worst path >= 333 ns against 250 ns, hot path +116 % against +10 %, a new nested loop with no derivable trip count, recursion from `std::sort`, frame 320 B against 288 B |
| the budget's anchor | the 2,500 us scheduler slice | NVIDIA's 120 FPS / 8.3 ms per frame and the ~2 ms derived parser cost |
| the blind spot | SITL runs on x86 without the H7's scheduler stack | the test feeds 5 rows; a frame has 25,200, and NMS is quadratic in them |

## What LOCI measured (this machine, 2026-11-03)

Target `neoverse-v2` at 2601 MHz — see CONTRACT.md for what the figures are
and are not. All timing is the function's own blocks, one trip through each
loop whose count is not derivable (floors, `>=`), callee-excluded; the
before/after pairs are at object scope, the standalone baseline on the linked
test ELF (same parser code: the TU assembles identically with and without SVE).

| | upstream (baseline) | innocent edit | fix |
|---|---|---|---|
| `NvDsInferParseCustomYoloV7` worst path | >= 139 ns (ELF) / >= 172 ns (object) | **>= 333 ns (+94 %)** | >= 180 ns (+4.9 %) |
| `NvDsInferParseCustomYoloV7` hot path | >= 79 ns / >= 84 ns | **>= 182 ns (+116 %)** | >= 58 ns (path re-selected, -32 %) |
| `NvDsInferParseCustomYoloV7` energy, hot path | 111 nWs | 255 nWs | 81 nWs |
| loops in `NvDsInferParseCustomYoloV7` (derived trip counts) | 2 (row loop `?`, class loop 80) | **8 (7 `?`)**, j-loop at :260 nested in the i-loop at :259 | 2 (row loop `?`, class loop 80) |
| own stack frame | 224 B | **320 B** | 240 B |
| recursion cycles in the unit | 0 | **1** (`std::__introsort_loop`) | 0 |
| `NvDsInferParseCustomYoloV4` worst path | >= 174 ns / >= 215 ns | >= 216 ns | >= 216 ns |
| unit test (`tests/test_parser.cpp`) | pass | pass | pass |
| host sanity bench, 25,200-row frame (Apple M-series, not a Jetson or LOCI figure) | 1.4 ms | **148 ms** | 1.2 ms |

Envelopes: `results/baseline/`, `results/scenario/innocent/`,
`results/scenario/fix/`; the earlier pass at the generic `aarch64`
(Cortex-A53 model, 1728 MHz) is kept under `results/archive-aarch64-a53/`. The
contract and the reasoning behind each bound: `CONTRACT.md`; it is in force as
`.loci/contract.yaml`. `test.sh` asserts baseline PASS, innocent edit FAIL and
fix PASS or CAUTION; the recorded run (`results/test/20261110T121455Z/`) gave
PASS / FAIL / CAUTION, the fix's worst path sitting at 72 % of its 250 ns budget.

## Layout

- `src/nvdsparsebbox_Yolo.cpp` — NVIDIA's parser, byte-identical to the pinned
  upstream unless a scenario patch is applied (`third_party/YOLO_DEEPSTREAM_COMMIT`).
- `CMakeLists.txt`, `cmake/aarch64-neoverse-v2.cmake` — the one-TU cross build
  that writes `build/compile_commands.json` for LOCI's recipe;
  `cmake/aarch64-cortex-a78ae.cmake` is the Orin variant, for reference.
- `tests/test_parser.cpp` — the handful-of-boxes unit test; `tests/bench_parser.cpp`
  — the full-frame host sanity benchmark.
- `scenario/innocent_edit.patch`, `scenario/fix.patch`, `scenario/variants/` —
  the two edits as patches and as whole files; `apply.sh`, `revert.sh`.
- `.loci/build.yaml` — the LOCI recipe (machine-local, written by `loci init`);
  `.loci/contract.yaml` — the contract once accepted.
- `results/` — every LOCI envelope from the runs behind the numbers above.
- `third_party/` — pinned upstream clone and the fetched headers (git-ignored),
  `NOTICE` with their licenses.

## Demo flow (stage script)

1. **Baseline**: show `NvDsInferParseYoloV7` (40 lines) and the NVIDIA FPS
   table. Show the contract: six sentences. LOCI baseline run: green.
2. **The request**: paste the prompt from `PROMPTS.md`. "Two real bugs and the
   feature the README says is missing — a reviewer approves it."
3. **Tests say go**: `test_parser` passes natively and as an aarch64 binary.
4. **LOCI post-edit**: worst path >= 333 ns against 250 ns (FAIL), hot path
   +116 % against +10 % (FAIL), 7 loops without a derivable trip count where
   there was 1, `std::sort` recursion (FAIL), 320 B frame against 288 B (FAIL),
   the rule on nested candidate loops broken. FLAG: not shippable.
5. **The fix**: threshold per row, NMS left to `cluster-mode=2`. LOCI: PASS,
   worst path +4.9 %. Optional: the host bench shows 148 ms vs 1.2 ms per frame,
   on a laptop core, for what the five-row test could not see.
6. **Punchline**: NVIDIA's own table already shows this parser throttling the
   pipeline at INT8 on Orin. The contract quotes NVIDIA's numbers, LOCI enforces
   them on the Thor-class binary, and nothing had to run on a Jetson to say no.
