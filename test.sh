#!/usr/bin/env bash
# End-to-end check of the demo. Non-zero exit on any deviation.
#
#   ./test.sh            # full run (needs: bootstrap.sh done, loci signed in,
#                        #   .loci/build.yaml recorded, .loci/contract.yaml accepted)
#   ./test.sh --no-loci  # build + unit tests only (no LOCI calls, no metering)
#
# Steps:
#   1. build the aarch64 (neoverse-v2-tuned) object/.so and the test binaries (build.sh)
#   2. run the unit test natively and as an aarch64 binary in an arm64 container
#   3. LOCI baseline: prepare + measure the two parse entry points (metered),
#      judged against .loci/contract.yaml -> must PASS
#   4. apply scenario/innocent_edit.patch, unit test must still pass,
#      LOCI post-edit comparison -> must FAIL (a severity: fail bound breached)
#   5. apply scenario/fix.patch, unit test must pass,
#      LOCI post-edit comparison -> must PASS
#   6. revert to upstream and rebuild
#
# Every LOCI envelope is kept under results/test/<timestamp>/.
# The turn ids are script-supplied opaque tokens (the headless/CI form the LOCI
# house rules allow); in an interactive Claude Code session the hooks stamp them.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
NO_LOCI=0; [ "${1:-}" = "--no-loci" ] && NO_LOCI=1
FN="NvDsInferParseCustomYoloV7,NvDsInferParseCustomYoloV4"
SRC="$HERE/src/nvdsparsebbox_Yolo.cpp"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$HERE/results/test/$STAMP"; mkdir -p "$OUT"
fail() { echo "test.sh: FAIL: $*" >&2; exit 1; }
step() { echo; echo "== $*"; }

# JSON field reader for the envelopes (python3 -I: no site/user imports).
jf() { python3 -I -c 'import json,sys
d=json.load(open(sys.argv[1]))
for k in sys.argv[2].split("."):
    d=d.get(k) if isinstance(d,dict) else None
print("" if d is None else d)' "$1" "$2"; }

run_unit_tests() {
  ./build.sh tests >/dev/null
  ./build/tests/test_parser_host >/dev/null || fail "host unit test failed ($1)"
  if docker info >/dev/null 2>&1; then
    docker run --rm --platform linux/arm64 -v "$HERE/build/tests:/t:ro" ubuntu:24.04 /t/test_parser_aarch64 >/dev/null \
      || fail "aarch64 unit test failed in the arm64 container ($1)"
    echo "   unit test: pass (host and aarch64 container) [$1]"
  else
    echo "   unit test: pass (host only; docker not reachable) [$1]"
  fi
}

# loci_measure <tag> <turn> [--source]  -> writes $OUT/<tag>.{prepare,measure}.json, prints verdict
loci_measure() {
  local tag="$1" turn="$2" mode="${3:-}" pre="$OUT/$1.prepare.json" mea="$OUT/$1.measure.json" code
  if [ "$mode" = "--source" ]; then
    loci analyse prepare --project-root "$HERE" --source "$SRC" --phase post-edit --turn "$turn" \
      --caller loci-post-edit --signals hot_path_time,worst_path_time,energy > "$pre" 2>"$OUT/$tag.prepare.stderr" || true
  else
    loci analyse prepare --project-root "$HERE" --functions "$FN" --signals hot_path_time,worst_path_time \
      --turn "$turn" --caller exec-trace > "$pre" 2>"$OUT/$tag.prepare.stderr" || true
  fi
  [ "$(jf "$pre" ok)" = "True" ] || fail "loci analyse prepare ($tag): $(jf "$pre" error.message)"
  local id; id="$(jf "$pre" data.id)"
  set +e
  loci analyse measure --prepared "$id" --project-root "$HERE" --context-file "$CTX" --energy \
    --caller "$([ "$mode" = "--source" ] && echo loci-post-edit || echo exec-trace)" > "$mea" 2>"$OUT/$tag.measure.stderr"
  code=$?
  set -e
  [ "$(jf "$mea" ok)" = "True" ] || fail "loci analyse measure ($tag): $(jf "$mea" error.message) (exit $code)"
  [ "$(jf "$mea" data.contract)" = "project" ] || fail "measure ($tag) was not judged by the project contract"
  local v; v="$(jf "$mea" data.verdict)"
  echo "   LOCI verdict [$tag]: $v (exit $code)"
  python3 -I - "$mea" <<'EOF'
import json,sys
d=json.load(open(sys.argv[1]))["data"]
for j in d.get("judgements",[]):
    print("     %-7s %-55s observed=%s bound=%s" % (j.get("verdict","").upper(), (j.get("text") or "")[:55], j.get("observed"), j.get("bound")))
for fn,p in d.get("paths",{}).items():
    if "NvDsInfer" not in fn: continue
    h=p.get("hot_path",{}); w=p.get("worst_path",{})
    print("     %s: hot %s -> %s ns, worst %s -> %s ns (floors: %s)" % (fn, h.get("before_ns"), round(h.get("ns") or 0), w.get("before_ns_worst"), round(w.get("ns_worst") or 0), h.get("lower_bound")))
EOF
  LAST_VERDICT="$v"; LAST_CODE="$code"
}

step "1. build"
./scenario/revert.sh >/dev/null
./build.sh >/dev/null
echo "   object: $(ls build/CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/src/*.o)"

step "2. unit tests on the upstream parser"
run_unit_tests upstream

if [ "$NO_LOCI" = 1 ]; then echo; echo "test.sh: build and unit tests OK (--no-loci)"; exit 0; fi

step "3. LOCI preconditions"
command -v loci >/dev/null || fail "loci not on PATH"
PJ="$OUT/project.json"; loci project --project-root "$HERE" > "$PJ" 2>/dev/null || true
[ "$(jf "$PJ" ok)" = "True" ] || fail "loci project: $(jf "$PJ" error.message) — run /loci:init (INSTRUCTIONS.md step 3)"
CTX="$(jf "$PJ" data.context_file)"
CS="$OUT/contract-show.json"; loci contract show --project-root "$HERE" > "$CS" 2>/dev/null || true
if [ "$(jf "$CS" data.exists)" != "True" ] || [ "$(jf "$CS" data.counts.enabled)" = "0" ]; then
  echo "test.sh: no contract is in force. Draft it per INSTRUCTIONS.md step 5 and apply it with:" >&2
  echo "    loci contract accept --project-root \"$HERE\"" >&2
  exit 3
fi
echo "   recipe: $(jf "$PJ" data.recipe) target $(jf "$PJ" data.loci_target); contract: $(jf "$CS" data.counts.enabled) enabled entries"

step "4. LOCI baseline (upstream parser) — must PASS"
loci_measure baseline "test-$STAMP-baseline"
[ "$LAST_VERDICT" = "pass" ] || fail "baseline verdict is '$LAST_VERDICT', expected pass"

step "5. innocent edit — unit test must pass, LOCI must FAIL"
T="test-$STAMP-innocent"
loci build snapshot --source "$SRC" --project-root "$HERE" --turn "$T" > "$OUT/innocent.snapshot.json" 2>/dev/null || true
./scenario/apply.sh innocent >/dev/null
run_unit_tests innocent
loci_measure innocent "$T" --source
[ "$LAST_VERDICT" = "fail" ] || fail "innocent-edit verdict is '$LAST_VERDICT', expected fail"
[ "$LAST_CODE" = "2" ] || fail "innocent-edit measure exit code is $LAST_CODE, expected 2 (severity: fail breach)"

step "6. fix — unit test must pass, LOCI must PASS"
./scenario/revert.sh >/dev/null
T="test-$STAMP-fix"
loci build snapshot --source "$SRC" --project-root "$HERE" --turn "$T" > "$OUT/fix.snapshot.json" 2>/dev/null || true
./scenario/apply.sh fix >/dev/null
run_unit_tests fix
loci_measure fix "$T" --source
[ "$LAST_VERDICT" = "pass" ] || fail "fix verdict is '$LAST_VERDICT', expected pass"

step "7. reset"
./scenario/revert.sh >/dev/null
./build.sh object >/dev/null
echo
echo "test.sh: OK — baseline PASS, innocent edit FAIL, fix PASS. Envelopes under $OUT"
