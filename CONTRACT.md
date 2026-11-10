# LOCI Contract — the budgets the parser is judged against

`.loci/contract.yaml` is where the requirements live. Once it is in force, every
LOCI measurement of this project is judged against it: a run closes on
**FAIL — measured X against your stated Y**, quoting the sentence back, instead
of a bare number. That is the beat of the demo: a budget the team wrote, enforced
on a cost that is invisible in the diff and in the unit test.

Target: LOCI `neoverse-v2` (Jetson Thor's Neoverse V3AE modeled as V2) at
2601 MHz, the Thor T5000 default 120 W `nvpmodel` CPU clock.

## The entries

| # | Sentence (verbatim in the file) | Signal | Bound | Baseline | Innocent edit | Fix |
|---|---|---|---|---|---|---|
| 1 | NvDsInferParseCustomYoloV7 worst-path time must not exceed 250 ns | `worst_path_time` | 250 ns | >= 139 ns (ELF) / >= 172 ns (object) | **>= 333 ns, 133 %** | >= 180 ns, 72 % |
| 2 | NvDsInferParseCustomYoloV7 hot-path time must not regress more than 10 % | `hot_path_time` regression | +10 % vs the pre-edit object | 84 ns | **>= 182 ns, +116 %** | >= 58 ns, -32 % |
| 3 | NvDsInferParseCustomYoloV7 must use no more than 288 bytes of stack for its own frame | `stack_frame_size` | 288 B | 224 B | **320 B** | 240 B |
| 4 | The parser library must not contain recursion: recursion cycles must stay at 0 | `recursion_cycles` invariant | 0 | 0 | **1** (`std::__introsort_loop` from `std::sort`) | 0 |
| 5 | NvDsInferParseCustomYoloV7 must not run a loop over the candidate rows inside another loop over the candidate rows; the only nested loop allowed is the 80-class argmax inside the row loop | text-only rule, judged by the agent on the CFG | — | 2 loops (row loop `?`, class loop 80) | **8 loops, j-loop at :260 nested in the i-loop at :259** | 2 loops |
| 6 | NvDsInferParseCustomYoloV4 worst-path time must not exceed 320 ns | `worst_path_time` | 320 ns | >= 174 ns (ELF) / >= 215 ns (object) | >= 216 ns | >= 216 ns |

All computable entries carry `severity: fail` (the sentences say "must"). The
figures are LOCI's, from `results/` (see STATUS.md for which envelope each one
comes from). Expected verdicts: baseline PASS, innocent edit FAIL (entries 1, 2,
3, 4 breached, 5 broken), fix PASS. Those are a reading of the measured figures
against the drafted bounds; the computed verdict appears the first time LOCI
runs with the contract in force.

## How the sentences reach the file

The LOCI plugin drafts and the repository owner applies — a bound nobody applied
is one nobody set. This folder ships the six sentences as a pending draft
(`loci contract draft show --project-root .`; for a fresh clone,
`loci contract draft add --project-root . < results/contract/draft-entries.json`),
produced from the measurements above. To put them in force:

```
loci contract accept --project-root /path/to/deepstream-yolo-parser
git add .loci/contract.yaml
```

Or re-state any of them in your own words to `/loci:contract` in a Claude Code
session; the skill drafts the entry and hands the same `accept` line back.

## Why these numbers

**What LOCI's per-call figure is, and is not.** `NvDsInferParseCustomYoloV7`
walks `rows = numElements / d[1]` rows; at 640x640 a YOLOv7 head emits 25,200.
That count is a run-time value, so LOCI derives no trip count for the row loop
(`trips ?`, "limit register w21 is written inside the loop") and reports the
path through **one** trip of each such loop as a floor (`>=`). The 80-class
argmax inside it is derived exactly (`trips 80`, `init x0=0; step +1; limit 80`).
Every timing figure is the function's own blocks at the recipe's clock,
callee-excluded: `clamp`, `std::vector::push_back` reallocation and
`operator new` bodies are not in it. It is a path estimate on LOCI's
`neoverse-v2` model (trained at 2.8 GHz) scaled linearly to 2601 MHz — not a
bench measurement, and the Neoverse V3AE in Thor is modeled by its V2
predecessor, the nearest core LOCI has. Nothing in this file multiplies a
floor by 25,200.

**The frame budget the bounds sit inside.** NVIDIA's own table for this sample
on a Jetson AGX Orin 64GB (MAXN, 1080p, YOLOv7) is the anchor
([yolo_deepstream README, Performance](https://github.com/NVIDIA-AI-IOT/yolo_deepstream#performance)):

| precision | streams | deepstream-app FPS, cuda post-process | deepstream-app FPS, cpu post-process |
|---|---|---|---|
| FP16 | 1 | 124 | 120 |
| FP16 | 16 | 145 | 135 |
| INT8 | 1 | 175 | 128 |
| INT8 | 16 | 264 | 135 |

At the published 120 FPS the whole pipeline has 8.33 ms per frame. Subtracting
frame times, the INT8 single-stream pair (1/128 s - 1/175 s) attributes about
2.1 ms per frame to the CPU post-process path (host copy included) — a derived
figure, not a published one. The parser is already the bottleneck once the GPU
is fast; that is why its cost per call is worth a fence. Thor's faster CPU
moves the absolute numbers, not the shape of the problem.

**Entry 1 (250 ns).** The pristine parser's worst-path floor is 139 ns on the
linked test ELF and 172 ns on the object LOCI compiles for a post-edit
comparison. 250 ns is a round number 1.45-1.8x the floor: enough headroom that
an honest one-line fix (the threshold compare adds 4.9 %) stays well inside, and
low enough that one more loop body on the per-row path (the innocent edit's
333 ns floor) does not. It is a fence on LOCI's per-call floor, not a claim that
250 ns x 25,200 rows fits the frame. (At the archived generic-aarch64/A53
target the same logic gave 700 ns over a 470-486 ns floor.)

**Entry 2 (+10 %).** The regression entry needs no absolute: it compares this
edit's hot path with the turn's pre-edit object. It is the entry that fires on
any edit, whatever the baseline drifts to. The fix moves the hot path to the
row-rejected branch (-32 %), which a regression bound rightly ignores.

**Entry 3 (288 B).** The function's own frame is the one stack signal LOCI can
compute on a relocatable object (worst-case *depth* needs a linked image —
STATUS.md explains why that run is not in this folder). The pristine frame is
224 B; 288 B leaves one more callee-saved pair and a spill slot. The innocent
edit's `std::sort`, two extra vectors and the NMS bookkeeping push it to 320 B.
Frames do not depend on the timing model; they are the same as at the A53 pass.

**Entry 4 (recursion 0).** `std::sort` is introsort; `std::__introsort_loop`
recurses on one partition. Its depth is bounded (`2*log2(n)` from `std::sort`
in `bits/stl_algo.h`), so an `unbounded_recursion` invariant would not fire —
`recursion_cycles` counts bounded cycles too, and this parser had none. On an
object this entry can only be breached, never satisfied, which is the honest
scope: the FAIL shows on the innocent edit; the fix's run leaves it unjudged.

**Entry 5 (the rule).** Text-only, judged by the agent from the control-flow
graph LOCI cuts. It is worded about what is fed to a pairwise loop, so a parser
that thresholds first and then runs NMS on the survivors in a helper still
breaks it only if the helper is inlined back into the row path — see PROMPTS.md.

**Entry 6 (320 ns).** The YOLOv4 path gets a bound that every variant passes
(174-216 ns, 68 % of the bound): the `maxIndex >= 0` guard is free, and a demo
contract should have a green row beside the red ones so the red one is
obviously the change.

## Two things to get right

1. **Bound the stable function, not the new one.** `nmsYoloV7` does not exist
   until the agent writes it; bounding `NvDsInferParseCustomYoloV7` is what
   catches a cost added below it, including one the compiler inlines into it.
2. **Do not over-tighten.** The baseline stays comfortably green (72 % / 68 % of
   its budgets after the fix), so the red row is the edit, not a stingy limit.
