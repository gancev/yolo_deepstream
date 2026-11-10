# STATUS — what is verified, what is not (2026-11-03, macOS arm64 host)

Second pass: the LOCI half was redone at the `neoverse-v2` target (Jetson Thor
T5000, Neoverse V3AE modeled as V2, 2601 MHz) on the user's decision. The first
pass at the generic `aarch64` target (Cortex-A53 model, 1728 MHz, AGX Orin
framing) is archived under `results/archive-aarch64-a53/`; everything outside
the LOCI half (sources, patches, tests, bootstrap) is unchanged.

## Pins

| What | Value |
|---|---|
| upstream | https://github.com/NVIDIA-AI-IOT/yolo_deepstream @ `e9b75770ea58713d1cb3902d67c36e11acb888d7` (2024-09-25, "update the model download link") |
| parser file | `deepstream_yolo/nvdsinfer_custom_impl_Yolo/nvdsparsebbox_Yolo.cpp`, MIT header, copied to `src/` |
| DeepStream headers | `nvcr.io/nvidia/deepstream:8.0-samples-multiarch` linux/arm64, digest `sha256:e76cb9a0f65f4dd8e4fa3560a49e1274107e837711be0c03d3c91ba2a72c8d99` (12.8 GB); `/opt/nvidia/deepstream/deepstream/sources/includes`, 68 files |
| TensorRT headers | `libnvinfer-headers-dev`, `libnvonnxparsers-dev` 10.13.2.6-1+cuda13.0 arm64 (matches the image's `libnvinfer10`) from developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa |
| CUDA headers | `cuda-cudart-dev-13-0` 13.0.96-1, `cuda-crt-13-0` 13.0.88-1 arm64, same repo |
| cross compiler | `/opt/homebrew/bin/aarch64-linux-gnu-g++` GCC 15.2.0 (Homebrew `aarch64-unknown-linux-gnu`, glibc sysroot, links static executables) |
| TU flags (LOCI object) | `-O2 -g -mcpu=neoverse-v2 -std=c++11 -fPIC -Wall -Wno-error=deprecated-declarations` via `cmake/aarch64-neoverse-v2.cmake` |
| test binaries | same, with `-mcpu=neoverse-v2+nosve` (Apple silicon has no SVE; the parser TU's assembly is identical with and without SVE, the SVE instructions were in the test harness) |
| host compiler | Apple clang 21 (`/usr/bin/clang++`), unit test + bench only |
| cmake | 4.4.0, Unix Makefiles generator |
| loci CLI | 0.2.126 (plugin 0.3.61 pins 0.2.126) |
| LOCI recipe | target `neoverse-v2`, validated `replay-compare` (LOCI's compile byte-identical to the CMake object), clock 2601 MHz scaled from the model's 2.8 GHz, `confirmed_by_user: false` |
| git | folder `git init`-ed, nothing committed (the owner commits after `contract accept`) |

## Verified end to end on this machine

- `bootstrap.sh` steps as individual commands (clone, image pull, header
  extraction, apt download + `dpkg -x` in an arm64 `ubuntu:24.04` container).
  The script was assembled from those commands and syntax-checked, not re-run
  from an empty folder (every step is idempotent and skips existing outputs).
- `build.sh`: aarch64 relocatable object with debug info, `.so`, static aarch64
  `test_parser_aarch64` / `bench_parser_aarch64`, host `test_parser_host` /
  `bench_parser_host`; `build/compile_commands.json` with one entry
  (`-mcpu=neoverse-v2`).
- Unit test passes on the upstream parser, the innocent edit and the fix, both
  natively and as an aarch64 binary inside `ubuntu:24.04` (linux/arm64). A
  plain `-mcpu=neoverse-v2` test binary dies with SIGILL (exit 132) in that
  container, hence `+nosve` for the test executables only.
- `loci init --refresh --target=neoverse-v2 --clock=2601MHz` (the first pass's
  `aarch64` recipe was replaced; init reported it `recipe_stale` after the
  cmake files changed, and re-derived). `loci project` and `loci doctor` healthy.
- Baseline: `loci analyse prepare` + `measure` (exec-trace, `--energy`,
  `--select p6`/`p15` because the ranked proposals were the early-return
  guards) on the recipe's linked `build/tests/bench_parser_aarch64` (freshness
  `unverified`: one glibc source path from the toolchain build is not on this
  machine); `cfg` on both entry points; own frames with `--elf` the CMake
  object; agent verdicts recorded. `results/baseline/`.
- Innocent edit: `loci build snapshot` (source captured), `scenario/apply.sh
  innocent`, post-edit `prepare` (Before reconstructed from the captured
  source, object scope) + `measure`; `cfg`; the two escalations the envelope
  listed (`stack-depth`, `memory-report`) with `--parent-run`; recursion
  classified bounded (`std::__introsort_loop`, `depth_limit` from `std::sort`);
  all runs recorded and settled. `results/scenario/innocent/`.
- Fix: same flow against the pristine Before; `cfg`; stack escalation;
  recorded. `results/scenario/fix/`.
- Contract: six entries drafted with `loci contract draft add`, `lint --draft`
  clean, pending in `.loci/build/contract.draft.yaml` (copy in
  `results/contract/contract.draft.yaml`, entries in
  `results/contract/draft-entries.json`).
- `test.sh --no-loci` passes. `test.sh` runs steps 1-3 and stops at the
  contract gate with exit 3 and the `accept` line, by design.
- Host sanity benchmark (`tests/bench_parser.cpp`, 25,200 rows, Apple M-series):
  upstream 1.41 ms, innocent 147.6 ms, fix 1.24 ms per call
  (`results/scenario/host-bench.txt`; the first pass measured 1.05 / 219 / 0.71 ms
  — host noise, same ratio class). Not a Jetson or LOCI figure.

## Measured figures (LOCI, neoverse-v2 at 2601 MHz, callee-excluded, floors)

| run | artifact | `NvDsInferParseCustomYoloV7` hot / worst / energy | `NvDsInferParseCustomYoloV4` hot / worst | envelope |
|---|---|---|---|---|
| baseline (exec-trace) | linked ELF | >= 79.1 ns / >= 139.4 ns / 110.8 nWs (worst 176.4 nWs) | >= 66.0 ns / >= 173.8 ns | `results/baseline/measure.json`, manifest `m-b350c6b98988` |
| baseline Before (object, from the post-edit pairs) | object | 84.2 ns / 171.6 ns | 77.7 ns / 215.0 ns | `before_ns` fields in the scenario envelopes |
| innocent edit (post-edit) | object | >= 182.0 ns (+116 %) / >= 332.9 ns (+94 %) / 254.9 nWs | >= 78.7 ns / >= 215.8 ns | `results/scenario/innocent/measure.json`, `m-d34ee5bd5dd4` |
| fix (post-edit) | object | >= 57.5 ns (re-selected, -32 %) / >= 180.0 ns (+4.9 %) / 80.5 nWs | >= 78.7 ns / >= 215.7 ns | `results/scenario/fix/measure.json`, `m-642d02ea2866` |

Archived first pass (generic `aarch64`, Cortex-A53 model, 1728 MHz): V7
baseline >= 313 / >= 470 ns (ELF), 293 / 486 ns (object); innocent >= 496 /
>= 829 ns; fix >= 189 / >= 503 ns. Same loop shapes, same frames.

Structure: baseline V7 2 loops (row loop `?`, class loop 80 exact); innocent
8 loops (7 `?`), j-loop :260 nested in i-loop :259, plus the `std::sort`
templates with `std::__introsort_loop` recursion; fix 2 loops. Own frames V7:
224 B / 320 B / 240 B; V4: 272 B / 256 B / 256 B. 3 indirect call sites
(`blr x2`) in every variant, all in the `std::endl` path of the
num-classes-mismatch warning. Object-scope walk V7: 288 B / 768 B / 304 B.

## Not verified / known limitations

- **Verdicts.** No contract is in force (the plugin drafts, the owner applies).
  Every run above reports `contract: none`; the PASS / FAIL / PASS in
  README.md and CONTRACT.md are the drafted bounds read against the measured
  figures, not CLI-computed verdicts. `test.sh` steps 4-7 (the assertions) have
  therefore not executed. First thing after `accept`: `./test.sh`.
- **Timing model.** LOCI's `neoverse-v2` target is a Neoverse V2 model; Thor's
  Neoverse V3AE is its successor and not a core LOCI models separately. The
  figures are consistent with each other (the deltas are the demo), not
  calibrated absolutes for V3AE. The Orin alternative (A78AE at 1728 MHz) is a
  clock change on the same V2 model, indicative only.
- **Whole-binary stack depth.** `loci analyse stack` on the statically linked
  glibc test binary ran for more than 30 minutes without finishing in the first
  pass (the lifter logs `Unsupported Dirty arm64g_dirtyhelper_MRS_MIDR_EL1`
  from glibc's ifunc resolvers) and was stopped; not retried at this target.
  All stack figures are object scope: own frames, plus the single-TU walk. The
  contract's stack entry is therefore `stack_frame_size`, not `stack_depth`.
  `test.sh` never runs a whole-binary walk.
- **Floors, not per-frame costs.** The row loop's trip count is a runtime value
  (`rows = numElements / d[1]`), so every parser figure counts one row; no
  envelope and no document multiplies by 25,200. The 2.1 ms per-frame figure
  is derived from NVIDIA's (Orin) FPS table, not from LOCI; no Thor table exists.
- **Turn ids.** This folder was built by a subagent with no hook-stamped turn
  (`loci hook prompt-submit` stamped nothing; no `[loci] turn=` line), so the
  turn tokens are script-supplied (`demo-v2-<timestamp>`), the headless/CI
  form the house rules allow. `test.sh` does the same. In a live Claude Code
  session the hooks supply them.
- **Recorded artifact.** Init chose `build/tests/bench_parser_aarch64` among
  three linked programs; it is stale as soon as a patch is applied, and every
  post-edit run falls back to the object LOCI compiles (the intended scope for
  a before/after pair). `cmake --build build` (the recipe's `full_build`)
  rebuilds only the `.so`; `build.sh` rebuilds the test binaries.
- **Baseline ELF vs object.** The baseline exec-trace measured the linked ELF
  (139 ns worst) while the post-edit Before is the object (172 ns); both are
  reported. The parser code is the same in both (checked by diffing the
  compiler's assembly for the two flag sets); the difference is layout and
  relocation handling.
- **On-target.** Nothing ran on a Jetson Thor or Orin; the DeepStream `.so`
  link line with `-lnvinfer -lcudart` was not reproduced (the CPU-only `.so`
  links from the single object without them).
- **Headers.** The DeepStream samples image carries runtime libraries only;
  TensorRT and CUDA headers came from the apt repo (above). No NGC login was
  needed; the SDK tarball route was not tried.

## Timings on this machine

| step | wall time |
|---|---|
| `docker pull` deepstream 8.0 samples multiarch (arm64) | about 14 min |
| apt download + extract of the four header packages (arm64 container) | about 1 min |
| `build.sh` (object + .so + 4 test binaries) | 5 s |
| `loci analyse prepare` (compile + rank) | 20-60 s |
| `loci analyse measure` (metered) | 1-2 min |
| `loci analyse cfg` | 10-30 s |
| `loci analyse stack` on the object | 10-20 s; on the static ELF: aborted after >30 min |
| `test.sh --no-loci` | 6 s |

Metered calls spent: 3 on this pass (`measure` on baseline, innocent, fix), 3
on the archived pass. `test.sh` spends 3 per run.
