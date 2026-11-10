# INSTRUCTIONS — bootstrap to reset, copy-pasteable

All commands run from this folder on a macOS arm64 host with Homebrew, docker
(colima) and the LOCI CLI signed in. Linux x86 works the same with an aarch64
cross GCC and docker with binfmt for arm64.

## 0. Host tools (once)

```bash
brew install aarch64-unknown-linux-gnu cmake colima docker
colima start --arch aarch64              # docker daemon; arm64 containers run natively on Apple silicon
loci auth status                         # LOCI CLI on PATH and signed in (plugin: /loci:setup, then `loci login`)
```

## 1. Bootstrap (upstream source + headers)

```bash
./bootstrap.sh
```

Clones `NVIDIA-AI-IOT/yolo_deepstream` at `third_party/YOLO_DEEPSTREAM_COMMIT`,
pulls `nvcr.io/nvidia/deepstream:8.0-samples-multiarch` for linux/arm64
(12.8 GB) and copies `sources/includes` out, then downloads the matching
TensorRT 10.13 and CUDA 13.0 header packages (arm64) from NVIDIA's public apt
repo in a throw-away `ubuntu:24.04` container and extracts them. Nothing is
installed on the host. `DS_IMAGE`, `TRT_VER`, `CUDA_PKGS` override the versions.

## 2. Build

```bash
./build.sh                               # or: ./build.sh object | ./build.sh tests
./build/tests/test_parser_host           # native unit test
docker run --rm --platform linux/arm64 -v "$PWD/build/tests:/t:ro" ubuntu:24.04 /t/test_parser_aarch64
./build/tests/bench_parser_host 3        # optional: full-frame host sanity bench (built by build.sh tests)
```

Outputs: `build/CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/src/nvdsparsebbox_Yolo.cpp.o`
(aarch64, `-O2 -g -mcpu=neoverse-v2 -std=c++11 -fPIC`), `build/libnvdsinfer_custom_impl_Yolo.so`,
`build/compile_commands.json`, static aarch64 test binaries under `build/tests/` (built with `-mcpu=neoverse-v2+nosve` so they run in the arm64 container on Apple silicon, which has no SVE; `TEST_CPU=neoverse-v2` on a Thor/Grace box; the parser TU assembles identically either way).

## 3. LOCI init (records the recipe; machine-local)

In a Claude Code session opened in this folder, `/loci:init` asks the one
question (target) and the clock and writes `.loci/build.yaml`. The equivalent
headless line, with the answers this demo uses:

```bash
loci init --project-root "$PWD" --compdb="$PWD/build/compile_commands.json" --compdb-kind=generated \
          --target=neoverse-v2 --clock=2601MHz
loci project --project-root "$PWD"       # shows the recipe, target and context file
loci doctor
```

- `--target=neoverse-v2`: LOCI's Neoverse V2 model (trained at 2.8 GHz), the
  nearest model to the Jetson Thor T5000's Neoverse V3AE cores; the TU is
  compiled with `-mcpu=neoverse-v2` to match. The probe's other candidate,
  `aarch64`, is a Cortex-A53 model (the AGX Orin's Cortex-A78AE is not a core
  LOCI knows; `cmake/aarch64-cortex-a78ae.cmake` is kept for reference).
- `--clock=2601MHz`: the Jetson Thor T5000 default 120 W `nvpmodel` CPU clock.
  To approximate an AGX Orin (Cortex-A78AE at the 30 W default) with the same
  recipe, `loci frequency set 1728MHz --project-root "$PWD"` (2201.6 MHz for
  Orin MAXN); the core model stays V2, so treat those figures as indicative.
- Review and confirm the recipe yourself (the plugin never passes `--confirmed`
  on your behalf): `/loci:init` -> "Looks right", or
  `loci init --project-root "$PWD" --compdb="$PWD/build/compile_commands.json" --compdb-kind=generated --target=neoverse-v2 --clock=2601MHz --confirmed`.

## 4. Baseline (free + one metered call)

```bash
T="demo-$(date -u +%Y%m%dT%H%M%SZ)"          # turn token; the Claude Code hooks stamp one in a live session
CTX=$(loci project --project-root "$PWD" | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"]["context_file"])')
FN="NvDsInferParseCustomYoloV7,NvDsInferParseCustomYoloV4"
loci analyse prepare --project-root "$PWD" --functions "$FN" --signals hot_path_time,worst_path_time --turn "$T" --caller exec-trace
loci analyse measure --prepared <id from prepare> --project-root "$PWD" --context-file "$CTX" --energy --caller exec-trace   # metered
loci analyse cfg   --project-root "$PWD" --turn "$T" --caller control-flow --functions "$FN" --context-file "$CTX"
loci analyse stack --project-root "$PWD" --turn "$T" --caller stack-depth --entry-functions "$FN" --context-file "$CTX" \
                   --elf "$PWD/build/CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/src/nvdsparsebbox_Yolo.cpp.o"   # own frames; see STATUS.md on the linked ELF
```

In Claude Code the same is `/loci:exec-trace NvDsInferParseCustomYoloV7`,
`/loci:control-flow NvDsInferParseCustomYoloV7`, `/loci:stack-depth`.

## 5. Contract (drafted here, applied by you)

```bash
loci contract draft add --project-root "$PWD" < results/contract/draft-entries.json
loci contract lint --draft --project-root "$PWD"
loci contract draft show --project-root "$PWD"
loci contract accept --project-root "$PWD"        # writes .loci/contract.yaml; review, then git add it
loci contract show --project-root "$PWD"
```

The six sentences and their derivation are in CONTRACT.md. Or state them to
`/loci:contract` in your own words.

## 6. Scenario

```bash
# innocent edit
T="demo-innocent-$(date -u +%Y%m%dT%H%M%SZ)"
loci build snapshot --source "$PWD/src/nvdsparsebbox_Yolo.cpp" --project-root "$PWD" --turn "$T"   # arm the Before
scenario/apply.sh innocent
./build.sh tests && ./build/tests/test_parser_host                                                  # tests say go
loci analyse prepare --project-root "$PWD" --source "$PWD/src/nvdsparsebbox_Yolo.cpp" --phase post-edit \
     --turn "$T" --caller loci-post-edit --signals hot_path_time,worst_path_time,energy
loci analyse measure --prepared <id> --project-root "$PWD" --context-file "$CTX" --caller loci-post-edit   # exit 2 = FAIL
loci analyse cfg   --project-root "$PWD" --turn "$T" --caller control-flow --functions NvDsInferParseCustomYoloV7 --source "$PWD/src/nvdsparsebbox_Yolo.cpp" --context-file "$CTX"
loci analyse stack --project-root "$PWD" --turn "$T" --caller stack-depth --parent-run <id> --entry-functions "$FN" --source "$PWD/src/nvdsparsebbox_Yolo.cpp" --context-file "$CTX"

# fix
scenario/revert.sh
T="demo-fix-$(date -u +%Y%m%dT%H%M%SZ)"
loci build snapshot --source "$PWD/src/nvdsparsebbox_Yolo.cpp" --project-root "$PWD" --turn "$T"
scenario/apply.sh fix
./build.sh tests && ./build/tests/test_parser_host
loci analyse prepare ... --phase post-edit --turn "$T" --caller loci-post-edit --signals hot_path_time,worst_path_time,energy
loci analyse measure --prepared <id> ... --caller loci-post-edit                                      # exit 0 = PASS
```

In Claude Code: paste the prompt from PROMPTS.md and let the `loci-post-edit`
hook do all of this on the agent's edit.

## 7. Tests

```bash
./test.sh --no-loci     # build + unit tests (host and aarch64 container)
./test.sh               # + baseline PASS, innocent FAIL (exit 2), fix PASS; non-zero on any deviation
```

`test.sh` exits 3 with the `accept` line when no contract is in force. Every
envelope of a run lands under `results/test/<timestamp>/`.

## 8. Reset

```bash
scenario/revert.sh && ./build.sh object            # pristine source, fresh object/.so and compile database
loci contract draft clear --project-root "$PWD"    # drop a pending draft (the accepted file is yours to keep or delete)
loci build clean --project-root "$PWD"             # LOCI's own turn trees and dumps
rm -rf build third_party/deepstream-includes third_party/tensorrt-includes third_party/cuda-includes third_party/_debs third_party/yolo_deepstream   # full reset; bootstrap.sh refetches
```
