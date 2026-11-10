# Demo prompts — paste into Claude Code (session opened in this folder)

The scenario can be produced two ways:

- **As code (deterministic, what test.sh does):** `scenario/apply.sh innocent`
  applies the agent's edit as a patch; `scenario/apply.sh fix` applies the
  acceptable rewrite; `scenario/revert.sh` restores NVIDIA's file.
- **As a prompt (the live demo):** paste the prompt below and let the agent
  implement it in `src/nvdsparsebbox_Yolo.cpp`. The LOCI plugin's
  `loci-post-edit` skill fires on the edit and measures it against
  `.loci/contract.yaml`. With plan mode on, `loci-preflight` judges the plan
  before any code is written.

The prompt is an honest maintenance request. It names two real defects in the
file and the one feature the README says is missing; there is no hint of a
cost. The agent writes the plausible version; LOCI prices it on the Jetson Thor CPU model.

---

## The innocent request (stage prompt)

> In `src/nvdsparsebbox_Yolo.cpp` the YOLOv7 parser never applies the
> configured per-class threshold (`detectionParams.perClassPreclusterThreshold`),
> and the YOLOv4 path can index that vector with `maxIndex == -1` when a row's
> class scores are all zero. Fix the threshold handling in both paths, and add
> the non-maximum suppression the README says this post-processor is missing:
> suppress boxes of the same class that overlap a higher-confidence box by more
> than 0.65 IoU. Keep it self-contained in this file (no new dependencies) and
> keep the exported `NvDsInferParseCustomYoloV7` / `NvDsInferParseCustomYoloV4`
> signatures. `tests/test_parser.cpp` must still pass.

What the agent tends to write (and what `scenario/innocent_edit.patch`
contains): the `maxIndex >= 0` guard, confidence = objectness x best class score,
a greedy per-class NMS helper with a sort and a pairwise `i`/`j` loop, run over
**every decoded row** (all 25,200 of a 640x640 YOLOv7 head), and the threshold
applied to the NMS survivors afterwards. The unit test, a handful of boxes,
passes.

What LOCI reports (results/scenario/innocent/): a new loop nested inside the
candidate loop with no derivable trip count, the floor of the parser's own
path rising past the contract's worst-path and regression bounds, and the
structural rule on the parser broken. Verdict FLAG.

## The follow-up (after the FLAG)

> LOCI flagged the pairwise loop over all candidate rows. Apply the per-class
> pre-cluster threshold to each row before any further work, and leave the
> suppression to nvinfer's cluster stage — `config_infer_primary_yoloV7.txt`
> already sets `cluster-mode=2` with `nms-iou-threshold=0.65`, which runs on the
> few hundred rows that pass the threshold. Keep the `maxIndex` guard.

`scenario/fix.patch` is that rewrite. `tests/test_parser.cpp` still passes, and
LOCI's post-edit comparison closes on PASS against the same contract.

An alternative acceptable fix, if the parser must own NMS: threshold first, then
NMS on the survivors in a separate helper. Its per-call floor as seen by LOCI is
similar to the innocent edit's (the loop bodies are the same code), so it only
passes a contract whose rule is about **what is fed to the pairwise loop**, not
about the loop's existence. That is why the demo's contract words the rule the
way it does (see CONTRACT.md).

---

### Contract (set once; see CONTRACT.md for the numbers and their derivation)

```
NvDsInferParseCustomYoloV7 worst-path time must not exceed 250 ns
NvDsInferParseCustomYoloV7 hot-path time must not regress more than 10 %
NvDsInferParseCustomYoloV7 must use no more than 288 bytes of stack for its own frame
The parser library must not contain recursion: recursion cycles must stay at 0
NvDsInferParseCustomYoloV7 must not run a loop over the candidate rows inside another loop over the candidate rows; the only nested loop allowed is the 80-class argmax inside the row loop
NvDsInferParseCustomYoloV4 worst-path time must not exceed 320 ns
```

Type each sentence to `/loci:contract`; the skill drafts the entry and you
apply it with `! loci contract accept`.
