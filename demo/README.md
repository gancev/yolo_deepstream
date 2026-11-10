# Live demo — the parser on a camera, three versions, one keypress apart

A real-time window on macOS that runs NVIDIA's `NvDsInferParseCustomYoloV7`
on every frame from your camera (or a video, or a still image), and lets
you switch between three versions of it while the audience watches:

| key | version | source |
|---|---|---|
| `1` | NVIDIA upstream | `third_party/yolo_deepstream/.../nvdsparsebbox_Yolo.cpp` (pinned commit) |
| `2` | the AI agent's edit | `scenario/variants/innocent/nvdsparsebbox_Yolo.cpp` |
| `3` | the acceptable fix | `scenario/variants/fix/nvdsparsebbox_Yolo.cpp` |

Each one is the file exactly as it is in the repo, built into its own host
dylib with `parser_shim.cpp`. The shim builds the same `NvDsInferLayerInfo`
and `NvDsInferParseDetectionParams` that nvinfer passes in, and times only the
parser call.

## Pipeline per frame (the same stages as DeepStream's nvinfer)

```
camera -> letterbox 640x640 -> YOLO ONNX (onnxruntime, standing in for TensorRT)
       -> [25,200 x 85 tensor] -> NvDsInferParseCustomYoloV7 (C++, NVIDIA's code)
       -> cluster stage: pre-cluster-threshold 0.25, per-class NMS 0.65, topk 300
          (config_infer_primary_yoloV7.txt, cluster-mode=2)
       -> boxes on screen
```

The model is Ultralytics YOLOv5s v7.0 (FP16 ONNX). Its output is
`[1, 25200, 85]`: cx, cy, w, h, objectness, then 80 class scores. That is the
same tensor layout as YOLOv7 at 640x640, which is what the parser expects.

## Run it

```bash
./bootstrap.sh                    # once, if not done: NVIDIA headers
demo/setup.sh                     # once: venv (uv), model + sample image, parser dylibs
demo/run.sh                       # webcam 0
demo/run.sh --video talk.mp4      # a looped video
demo/run.sh --image demo/assets/sample.jpg   # no camera needed
demo/run.sh --fullscreen          # for the projector
demo/run.sh --ep cpu              # skip CoreML
demo/run.sh --image demo/assets/sample.jpg --selftest 15   # headless check
```

The first webcam run asks macOS for camera access for your terminal app. Allow
it, or grant it in System Settings > Privacy & Security > Camera.

Keys: `1` `2` `3` switch parser · `A` auto-cycle (8 s each, `--auto-seconds`)
· `SPACE` pause · `F` fullscreen · `Q` quit.

## What the screen shows

- **Video**: the boxes after the cluster stage. All three versions find the
  same objects, so the output alone gives no hint that anything is wrong.
- **Banner**: parser time for this frame, and the end-to-end FPS.
- **Per-frame bars**: inference, parser and cluster, measured on this Mac.
- **Timeline**: parser time over the last 200 frames on a log scale, coloured
  by the version that produced each frame. The amber line marks 8.3 ms, a whole
  frame at NVIDIA's published 120 FPS.
- **LOCI card**: LOCI's figures for the aarch64 object (Neoverse V2 model at
  2601 MHz, Jetson Thor class) from `results/`, set against the bounds drafted
  in `CONTRACT.md`. Below them is the line that gave the go-ahead: the 5-row
  unit test passes for all three versions.

Measured on an Apple M5 Pro (`--selftest 15`, `sample.jpg`):

| | inference | parser | parser output | boxes |
|---|---|---|---|---|
| upstream | ~20 ms | 0.66 ms | 25,200 rows (no threshold) | 5 |
| agent edit | ~20 ms | **276 ms** | 5 | 5 |
| fix | ~20 ms | 0.57 ms | 56 | 5 |

The agent's edit costs about 420 times more parser time per frame and gives
the same five boxes. The live FPS falls to about 3, from about 45.

## Binary view (key `B`): the before and after binaries

`B` replaces the video with the compiled **aarch64** code of
`NvDsInferParseCustomYoloV7`. The left side is always `upstream.o`; the right
side is the version picked with `1`/`2`/`3`. The objects are built by
`demo/build_binaries.sh`, which `setup.sh` runs, with the LOCI recipe's
compiler and flags (`aarch64-linux-gnu-g++ -O2 -g -mcpu=neoverse-v2`).
`loci elf diff` finds no changed functions between `upstream.o` and the CMake
object LOCI validated. The files differ only in the source paths recorded in
the debug info.

What it shows, all from LOCI's extracts:

- **Tiles**: instructions, basic blocks, loops, loops whose trip count can't be
  derived, and functions added (`loci elf asm`, `loci elf diff`).
  Agent edit: 247 -> 469 instructions, 2 -> 8 loops, 4 new functions.
  Fix: 247 -> 244 instructions, 2 -> 2 loops, 0 new functions.
- **Loops, before | after**: each loop with its source line and trip count.
  Red marks a loop with no trip count nested inside the row loop: the
  rows x rows NMS at `:259`/`:260`.
- **New functions**: the four `std::sort` internals (`__introsort_loop` is
  recursive).
- **Assembly, side by side**: aligned with difflib on the instruction shape.
  Red on the left means removed, green on the right means added. For the
  agent's edit it opens on the nested loop's header block. `[` `]` scroll,
  `N` / `P` jump to the next / previous change.

`demo/run.sh --binary` starts in this view. Files: `build/demo/aarch64/`
(`<variant>.o`, `<variant>-asm/` and `diff-<variant>/`).

## Side by side: without LOCI | with LOCI (`demo/compare.sh`)

Both shipped parsers run live at the same time on the same frames. The left
side is the agent's edit (shipped because its unit test passed). The right
side is the fix (shipped because LOCI flagged the edit). A capture thread
delivers frames in real time. Each side has its own pipeline thread and
always takes the newest frame, like DeepStream on a live source. When a
pipeline can't keep up, its frames drop.

```bash
demo/compare.sh                                    # window, sample video (OpenCV vtest.avi)
demo/compare.sh --camera 0                         # window, webcam
demo/compare.sh --record demo/out/compare.mp4 --seconds 30            # also record
demo/compare.sh --record demo/out/compare.mp4 --seconds 30 --no-window
```

`--record` writes `compare.mp4` (1920x1080, 30 fps, real time) plus
`compare-without-loci.mp4` and `compare-with-loci.mp4`, the two halves on
their own.

Each side shows:
- the NVIDIA parser's time for the frame on screen, large, on the picture
- the per-frame bar (inference | parser | cluster), on one 0-400 ms scale for
  both sides, with a marker at the source's frame interval
- parser median, parser share of the frame, FPS, latency and frames dropped
- a log-scale history of parser time

Measured on this Mac, 30 s of vtest.avi (10 fps):

| | parser median | parser share | FPS | latency | frames processed |
|---|---|---|---|---|---|
| without LOCI (agent's edit) | 361 ms | 93 % | 2.5 | ~460 ms | 78 of 302 |
| with LOCI (fix) | 0.57 ms | 2 % | 10 (all) | ~27 ms | 301 of 302 |

## Honesty notes for the talk

- The ms figures on screen are wall clock on this laptop, not Jetson
  figures. The LOCI card is the Jetson-class figure, read from the compiled
  aarch64 object without the device.
- The LOCI card is static. It shows the recorded runs in `results/`; the demo
  does not call LOCI per frame. Its PASS / FAIL is read against the drafted
  contract. After `loci contract accept`, `./test.sh` computes the verdicts
  for real.
- onnxruntime stands in for TensorRT, and a Python NMS stands in for nvinfer's
  cluster stage. The parser in the middle is NVIDIA's C++, unchanged.
