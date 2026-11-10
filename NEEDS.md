# NEEDS — what only the repository owner can supply

1. **Apply the contract.** The six entries are drafted (`loci contract draft
   show --project-root .`; the JSON is in `results/contract/draft-entries.json`
   for a fresh clone). The plugin may not run `accept`; you do:

   ```
   loci contract accept --project-root /Users/stojangancev/projects/aurorora/NVIDIA/deepstream-yolo-parser
   ```

   Until then every run reports `contract: none`, `test.sh` stops at step 3
   with exit 3, and the FAIL / PASS verdicts in CONTRACT.md and README.md are
   expected values read off the measured figures, not computed ones. After
   `accept`, run `./test.sh` once and the run's envelopes land under
   `results/test/`.

2. **Confirm the LOCI recipe.** `.loci/build.yaml` was written by
   `loci init --refresh` with the target (`neoverse-v2`) and clock (2601 MHz)
   you decided, but `confirmed_by_user` is false and every report says so. Review it in
   `/loci:init` ("Looks right") or re-run the init line in INSTRUCTIONS.md
   step 3 with `--confirmed`. Decide the recorded artifact while you are there:
   init chose `build/tests/bench_parser_aarch64` (a choice among three linked
   programs); `loci init set artifacts.elf=build/tests/test_parser_aarch64` if
   you prefer the test binary.

3. **Decide the deployed clock / board.** 2601 MHz is the Jetson Thor T5000
   default 120 W mode on LOCI's Neoverse V2 model (Thor's cores are Neoverse
   V3AE). For an AGX Orin (Cortex-A78AE, 1728 MHz at 30 W, 2201.6 MHz MAXN)
   `loci frequency set 1728MHz --project-root .` keeps the V2 core model and
   only rescales; the contract's ns bounds scale with it (172 ns at 2601 MHz is
   about 259 ns at 1728 MHz) and would need re-drafting. A Thor-specific
   (V3AE) or A78AE timing model is a LOCI product need, not a knob.

4. **A Jetson AGX Thor (or Orin) for on-target ground truth (optional).** Nothing here
   ran on a Jetson (Thor or Orin). To validate the direction of the result:
   build the .so with the upstream Makefile on JetPack 7.x (Thor) or 6.x (Orin) with DeepStream 8.0, switch
   `config_infer_primary_yoloV7.txt` to `parse-bbox-func-name=NvDsInferParseCustomYoloV7`
   and `disable-output-host-copy=0`, and compare `deepstream-app` FPS with the
   pristine parser, the innocent edit and the fix; or put NVTX ranges around
   `gst_nvinfer_output_loop()` under Nsight Systems. The upstream `-lnvparsers`
   link flag was removed from TensorRT 10; drop it from the Makefile.

5. **Whole-binary stack depth needs a linked artifact LOCI can walk in
   reasonable time.** The statically linked glibc test binary did not finish a
   whole-binary walk in 30 minutes (STATUS.md). Options: a dynamically linked
   test binary built against an aarch64 sysroot with shared libstdc++ (then
   libc calls become unknown callees, 64 B each), or the real
   `libnvdsinfer_custom_impl_Yolo.so` loaded into a tiny host program on a
   Jetson. Until then the stack signal in the contract is the function's own
   frame (`stack_frame_size`), which the object answers.

6. **LOCI quota.** The folder's runs used six metered `measure` calls in all
   (three on the archived aarch64/A53 pass, three on the neoverse-v2 pass). `test.sh` uses three per run.

7. **Decide whether to commit the headers' presence.** `third_party/*-includes`
   and `third_party/_debs` are git-ignored because they are NVIDIA EULA
   material (DeepStream SDK, TensorRT, CUDA). If the demo repo is private and
   the licenses allow it for your use, un-ignoring them makes a clone
   buildable without docker; otherwise keep `bootstrap.sh` as the fetch path.

8. **Git.** The folder was `git init`-ed but nothing is committed; commit when
   the contract is accepted so `.loci/contract.yaml` travels with the demo.
