# LOCI demo: numbers for every change, at the speed agents write code

AI agents now write code faster than anyone can review it line by line. A change
can fix real bugs, read well in review and pass its tests, and still cost
the product something nobody looked at. This demo shows what changes when every
edit arrives with measurements of the compiled code, judged against limits the
team wrote down.

The subject is NVIDIA's open-source DeepStream YOLO CPU parser,
`nvdsparsebbox_Yolo.cpp` from
[NVIDIA-AI-IOT/yolo_deepstream](https://github.com/NVIDIA-AI-IOT/yolo_deepstream).
It runs on the CPU after every TensorRT inference, once per video frame. The
target is a Jetson Thor-class CPU. No Jetson, GPU or NGC login is needed: LOCI
reads everything from the compiled aarch64 binary.

**Start here:**

- `docs/scenario-infographic.html` is the whole story on one page (download it
  and open it in a browser).
- `demo/out/compare.mp4` (30 s) shows two live pipelines on the same frames.
  On the left is "without LOCI": the agent's edit, shipped because its unit test
  passed. On the right is "with LOCI": the fix, shipped because LOCI flagged the
  edit. `compare-without-loci.mp4` and `compare-with-loci.mp4` are the two
  halves on their own.

## 1. The request

An agent is asked to "fix the threshold handling and add the missing NMS"
(the exact prompt is in `PROMPTS.md`). It produces a plausible change:

- it fixes the two real defects in NVIDIA's file;
- it adds non-maximum suppression inside the parser, run on all 25,200
  candidate rows before thresholding: a full `std::sort` plus a pairwise IoU loop.

The 5-box unit test (`tests/test_parser.cpp`) passes for NVIDIA's original, the
agent's change and the fix alike. Tests cannot tell them apart.

The agent's change is a prepared, representative patch
(`scenario/innocent_edit.patch`), not a recording of one specific agent run.
Pasting the prompt from `PROMPTS.md` into an agent session reproduces it live.

## 2. The contract: limits written before the change

`CONTRACT.md` explains each limit and where its number comes from (NVIDIA's own
published Orin frame-rate table: about 2.1 ms per frame is attributable to this
CPU parser at INT8). The contract has six sentences:

| # | Requirement |
|---|---|
| 1 | `NvDsInferParseCustomYoloV7` worst-path time must not exceed 250 ns |
| 2 | `NvDsInferParseCustomYoloV7` hot-path time must not regress more than 10 % |
| 3 | `NvDsInferParseCustomYoloV7` must use no more than 288 bytes of stack for its own frame |
| 4 | The parser library must not contain recursion: recursion cycles must stay at 0 |
| 5 | `NvDsInferParseCustomYoloV7` must not run a loop over the candidate rows inside another loop over the candidate rows |
| 6 | `NvDsInferParseCustomYoloV4` worst-path time must not exceed 320 ns |

The contract is in force: `.loci/contract.yaml`, accepted by the repository
owner with `loci contract accept`. Every LOCI measurement of this project is
judged against it. `./test.sh` checks the verdicts end to end; the run recorded
in `results/test/20261110T121455Z/` gave the verdicts below.

## 3. What LOCI measured

LOCI measured the compiled aarch64 object: Neoverse V2 model at 2601 MHz, the
Jetson Thor T5000's default CPU clock.

**Timing, judged by LOCI against the contract** (`./test.sh`, envelopes in
`results/test/20261110T121455Z/`):

| `NvDsInferParseCustomYoloV7` | NVIDIA original | agent's change | fix | limit |
|---|---|---|---|---|
| worst-path time | ≥ 172 ns | **≥ 333 ns: FAIL** | ≥ 180 ns: CAUTION | ≤ 250 ns |
| hot-path time | ≥ 84 ns | **≥ 182 ns, +116.1 %: FAIL** | ≥ 58 ns, −31.7 %: PASS | ≤ +10 % |
| **run verdict** | **PASS** | **FAIL** (exit 2) | **CAUTION** (exit 0) | |

The fix is within every limit. Its CAUTION is LOCI marking a figure at 70 % or
more of its limit: the worst path went from 69 % (172 ns) to 72 % (180 ns) of
the 250 ns budget. `test.sh` accepts PASS or CAUTION for the fix, because no
bound is breached. The original's baseline row in the test log reads 139 ns
because it is measured on the linked test binary. The 172 ns above is the same
function in the object, which the before/after comparisons use.

**Structure, from LOCI's stack and control-flow analysis** (envelopes in
`results/baseline/`, `results/scenario/innocent/` and `results/scenario/fix/`).
These checks back contract rules 3 to 5. This test run's timing check did not
judge them:

| `NvDsInferParseCustomYoloV7` | NVIDIA original | agent's change | fix | limit |
|---|---|---|---|---|
| own stack frame | 224 B | **320 B** | 240 B | ≤ 288 B |
| recursion cycles | 0 | **1** (`std::__introsort_loop`, from `std::sort`) | 0 | 0 |
| loops (no derivable trip count) | 2 (1) | **8 (7)**: loop at `:260` nested in the loop at `:259` | 2 (1) | no nested row loop |

"≥" means the figure is a floor. The row loop's length (25,200) is a run-time
value, so LOCI costs one trip through it. That is why the contract pairs its time
budgets with a structural rule: the nested row loop is what predicts the
blow-up, and LOCI names it at `src/nvdsparsebbox_Yolo.cpp:259-260`.

## 4. What it costs on a real frame

To check the direction and size of LOCI's findings, we ran a live pipeline on a
laptop (`demo/measure.py`, raw numbers in `demo/out/measurements.json`): an Apple
M5 Pro, 30 s of the OpenCV `vtest` clip per version, onnxruntime standing in for
TensorRT. Only the parser differs between runs.

| | NVIDIA original | agent's change | fix |
|---|---|---|---|
| parser time per frame (median) | 0.83 ms | **402 ms** | 0.65 ms |
| parser share of each frame | 2.4 % | **93 %** | 2.0 % |
| frames per second delivered | 10 | **2.3** | 10 |
| frames processed out of 300 | 300 | **68** | 300 |

The agent's change is 487 times slower on a real frame, and every
detection on screen still looks right. The fix is 21 % faster than NVIDIA's
original because it applies the threshold first and leaves NMS to DeepStream's
cluster stage.

## 5. Openness: the same measurement on a real upstream fix

The review also found a real latent bug in NVIDIA's file. The YOLOv4 path
can read `perClassPreclusterThreshold[-1]` when every class score in a row is
zero. We opened the two-line fix upstream:
[NVIDIA-AI-IOT/yolo_deepstream#70](https://github.com/NVIDIA-AI-IOT/yolo_deepstream/pull/70).

Even a safety fix has a cost, so the PR carries LOCI's measurement of it
(envelope: `results/upstream-pr-70/measure.json`):

| `NvDsInferParseCustomYoloV4` | before | after | change |
|---|---|---|---|
| typical path | ≥ 77.7 ns | ≥ 78.8 ns | +1.4 % |
| worst path | ≥ 215.0 ns | ≥ 215.8 ns | +0.4 % |

That's one extra compare-and-branch, and well inside entry 6's 320 ns limit.

## 6. Why the numbers matter

When code is generated in seconds, review shifts from "is this code right?" to
"what did this change do?". Tests answer the first question for the inputs they
cover. A measurement of the compiled code answers the second for every change: time per
path, stack, loops and recursion, each compared with the previous version and
with a limit someone chose. In this demo the agent's change and the fix differ
by more than a factor of 480 on a real frame. They pass the same unit test,
and on the numbers LOCI reports they are nothing alike.

## Limits of this evidence

- Nothing here ran on a Jetson. LOCI's figures are model estimates from the
  binary (Neoverse V2, the nearest core LOCI has to Thor's Neoverse V3AE, scaled
  from 2.8 GHz to 2601 MHz). The laptop figures show direction and order of
  magnitude.
- LOCI's times are per call, exclude the functions it calls, and cost each
  frame-dependent loop once. They do not convert into the laptop's milliseconds.
- The agent's change is a prepared patch.

## Reproduce it

See `INSTRUCTIONS.md` for the full sequence (host tools, `./bootstrap.sh`,
`./build.sh`, `loci init`, `loci contract accept`, `./test.sh`) and
`demo/README.md` for the live camera demo (`demo/setup.sh`, `demo/run.sh`).
