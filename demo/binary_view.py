"""Binary view for the live demo (key B): before/after compiled aarch64 code.

Reads what demo/build_binaries.sh wrote under build/demo/aarch64/: per variant,
LOCI's `elf asm` control-flow dump of NvDsInferParseCustomYoloV7 (blocks,
instructions, loops with derived trip counts) and `elf diff` against
upstream.o (functions added / modified). Before is always upstream.o; after
is the variant selected with 1/2/3.
"""
import difflib
import json
import os
import re
import subprocess

import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX
MONO = cv2.FONT_HERSHEY_PLAIN
TEXT = (235, 235, 235)
MUTED = (150, 150, 150)
CARD = (40, 37, 34)
LINE = (70, 66, 62)
GREEN = (0, 185, 118)
RED = (70, 70, 235)
AMBER = (0, 175, 255)
DEL_BG = (40, 40, 95)
INS_BG = (30, 75, 40)

LOOP_RE = re.compile(r"^\s+L(\d+)\s+loop\s+header (\S+)\s+latch (\S+)\s+depth (\d+)(?:\s+in (L\d+))?"
                     r"\s+blocks (\d+)\s+source=(\S+)\s+trips (\S+)")
BLOCK_RE = re.compile(r"^\[(\S+)_0x([0-9a-f]+)\] 0x([0-9a-f]+)-0x([0-9a-f]+)(.*)$")


def short_source(src):
    """'.../nvdsparsebbox_Yolo.cpp:259:260' -> ':259', 'stl_vector.h:392:392' -> 'stl_vector.h:392'."""
    first = src.split(";")[0]
    name, _, rest = first.rpartition("/")[2].partition(":")
    line = rest.split(":")[0] if rest else ""
    return f":{line}" if name.startswith("nvdsparsebbox_Yolo") else f"{name}:{line}"


def demangle(sym):
    raw = sym[sym.find("_Z"):] if "_Z" in sym else sym
    raw = raw.split(".")[0]
    try:
        out = subprocess.run(["c++filt", raw], capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        out = raw
    m = re.search(r"(std::[A-Za-z_]+)", out)
    return m.group(1) if m else out.split("(")[0][:40]


def load_variant(adir):
    """Parse LOCI's control-flow.txt into loops and an instruction listing."""
    cf = open(os.path.join(adir, "control-flow.txt")).read().splitlines()
    loops, lines, blocks = [], [], 0
    src = ""
    for ln in cf:
        m = LOOP_RE.match(ln)
        if m:
            loops.append({"id": "L" + m.group(1), "depth": int(m.group(4)), "parent": m.group(5),
                          "blocks": int(m.group(6)), "src": short_source(m.group(7)),
                          "trips": m.group(8)})
            continue
        m = BLOCK_RE.match(ln)
        if m:
            blocks += 1
            tag = m.group(5).strip()
            lines.append({"kind": "block", "text": f"bb_0x{m.group(2)}  {tag}", "addr": m.group(2), "norm": "#block"})
            continue
        if ln.startswith("source:"):
            src = short_source(ln[len("source:"):].strip())
            if lines and lines[-1]["kind"] == "block":
                lines[-1]["text"] += f"  {src}"
            continue
        if ln.startswith("  ") and not ln.startswith("  ->") and lines:
            ins = ln.strip()
            lines.append({"kind": "ins", "text": ins,
                          # branch targets and literal addresses differ between
                          # builds; compare the instruction shape only
                          "norm": re.sub(r"#-?0x[0-9a-f]+", "#imm", ins)})
    ins_count = sum(1 for l in lines if l["kind"] == "ins")
    return {"loops": loops, "lines": lines, "blocks": blocks, "instructions": ins_count}


def side_by_side(before, after):
    """Align two listings: rows of (left line|None, right line|None, tag)."""
    a = [l["norm"] for l in before["lines"]]
    b = [l["norm"] for l in after["lines"]]
    rows = []
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            for k in range(i2 - i1):
                rows.append((before["lines"][i1 + k], after["lines"][j1 + k], "equal"))
        else:
            n = max(i2 - i1, j2 - j1)
            for k in range(n):
                l = before["lines"][i1 + k] if i1 + k < i2 else None
                r = after["lines"][j1 + k] if j1 + k < j2 else None
                rows.append((l, r, op))
    return rows


class BinaryView:
    VARIANT_FILES = ("upstream", "innocent", "fix")

    def __init__(self, root):
        self.dir = os.path.join(root, "build/demo/aarch64")
        self.ok = all(os.path.exists(os.path.join(self.dir, f"{v}-asm", "control-flow.txt"))
                      for v in self.VARIANT_FILES)
        self.data, self.diff, self.rows, self.scroll = {}, {}, {}, {}
        if not self.ok:
            return
        for v in self.VARIANT_FILES:
            self.data[v] = load_variant(os.path.join(self.dir, f"{v}-asm"))
            self.data[v]["size"] = os.path.getsize(os.path.join(self.dir, f"{v}.o"))
        for v in ("innocent", "fix"):
            p = os.path.join(self.dir, f"diff-{v}", "diff.json")
            entries = json.load(open(p)) if os.path.exists(p) else []
            self.diff[v] = {
                "added": [demangle(e["symbol"]) for e in entries if e["status"] == "added"],
                "similarity": next((e.get("similarity_ratio") for e in entries
                                    if e["symbol"] == "NvDsInferParseCustomYoloV7"), None),
            }
            self.rows[v] = side_by_side(self.data["upstream"], self.data[v])
            nested = [l["id"] for l in self.data[v]["loops"] if l["depth"] >= 2 and l["trips"] == "?"]
            first = next((i for i, r in enumerate(self.rows[v])
                          if nested and r[1] and f"loop-header {nested[0]}," in r[1]["text"]), None)
            if first is None:
                first = next((i for i, r in enumerate(self.rows[v]) if r[2] != "equal"), 0)
            self.scroll[v] = max(0, first - 3)

    def scroll_by(self, key, n):
        if key in self.rows:
            self.scroll[key] = max(0, min(len(self.rows[key]) - 10, self.scroll[key] + n))

    def jump_next_change(self, key, back=False):
        rows = self.rows.get(key)
        if not rows:
            return
        cur = self.scroll[key] + 3
        rng = range(cur - 2, 0, -1) if back else range(cur + 1, len(rows))
        for i in rng:
            if rows[i][2] != "equal" and rows[i - 1][2] == "equal":
                self.scroll[key] = max(0, i - 3)
                return

    # ---------- drawing ----------

    def draw(self, canvas, x0, y0, w, h, variant):
        key = variant["key"]
        x1 = x0 + w
        cv2.rectangle(canvas, (x0, y0), (x1, y0 + h), (18, 17, 16), -1)
        if not self.ok:
            self._t(canvas, "Binary view: run demo/build_binaries.sh first", x0 + 30, y0 + 60, 0.7, AMBER, 2)
            return
        pad = 24
        x, y = x0 + pad, y0 + 34
        self._t(canvas, "COMPILED aarch64   NvDsInferParseCustomYoloV7", x, y, 0.62, TEXT, 2)
        self._t(canvas, "aarch64-linux-gnu-g++ 15.2  -O2 -mcpu=neoverse-v2 (the LOCI recipe's flags)   "
                "listing: loci elf asm   diff: loci elf diff", x, y + 22, 0.42, MUTED)
        before = self.data["upstream"]
        after = self.data[key]
        same = key == "upstream"

        # stat tiles
        y += 42
        def unk(d):
            return sum(1 for l in d["loops"] if l["trips"] == "?")
        nested_rows = [l for l in after["loops"] if l["depth"] >= 2 and l["trips"] == "?"]
        tiles = [
            ("instructions", before["instructions"], after["instructions"]),
            ("basic blocks", before["blocks"], after["blocks"]),
            ("loops", len(before["loops"]), len(after["loops"])),
            ("loops, trip count ?", unk(before), unk(after)),
            ("functions added", 0, len(self.diff.get(key, {}).get("added", []))),
        ]
        tw = (w - 2 * pad - 4 * 10) // 5
        for i, (name, b, a) in enumerate(tiles):
            tx = x + i * (tw + 10)
            cv2.rectangle(canvas, (tx, y), (tx + tw, y + 66), CARD, -1)
            self._t(canvas, name, tx + 10, y + 20, 0.42, MUTED)
            if same:
                self._t(canvas, f"{b}", tx + 10, y + 52, 0.8, TEXT, 2)
            else:
                worse = a > b
                col = RED if worse and (a - b) / max(b, 1) > 0.25 else (AMBER if worse else GREEN)
                self._t(canvas, f"{b}", tx + 10, y + 52, 0.62, MUTED, 1)
                bw = cv2.getTextSize(f"{b}", FONT, 0.62, 1)[0][0]
                self._t(canvas, "->", tx + 16 + bw, y + 52, 0.5, MUTED)
                self._t(canvas, f"{a}", tx + 44 + bw, y + 52, 0.8, col if a != b else TEXT, 2)

        # loops, before | after
        y += 92
        colw = (w - 2 * pad - 20) // 2
        loops_bottom = y
        for side, (title, d) in enumerate((("before: upstream.o", before),
                                           (f"after: {key}.o" if not same else "after: (select 2 or 3)", after))):
            cx = x + side * (colw + 20)
            self._t(canvas, title.upper(), cx, y, 0.45, MUTED)
            ly = y + 8
            for l in d["loops"]:
                ly += 22
                indent = 18 * (l["depth"] - 1)
                rows_in_rows = l in nested_rows and side == 1 and not same
                col = RED if rows_in_rows else (TEXT if l["trips"] != "?" or l["depth"] == 1 else AMBER)
                parent = f" in {l['parent']}" if l["parent"] else ""
                trips = "trips 80" if l["trips"] == "80" else f"trips {l['trips']}"
                self._t(canvas, f"{l['id']}{parent}", cx + indent, ly, 0.47, col, 1)
                self._t(canvas, f"src {l['src']}", cx + indent + 92, ly, 0.45, col)
                self._t(canvas, trips, cx + colw - 80, ly, 0.45, col)
            if side == 1 and not same:
                added = self.diff[key]["added"]
                ly += 30
                if added:
                    names = sorted({a.replace("std::", "") for a in added})
                    self._t(canvas, f"{len(names)} new functions, all std::sort internals:", cx, ly, 0.42, RED)
                    ly += 18
                    self._t(canvas, ", ".join(names), cx, ly, 0.4, RED)
                    ly += 18
                    self._t(canvas, "introsort_loop calls itself: recursion in the parser", cx, ly, 0.4, MUTED)
                if nested_rows:
                    ly += 18
                    self._t(canvas, "red loops: no trip count, nested in the row loop = rows x rows",
                            cx, ly, 0.4, RED)
                sim = self.diff[key]["similarity"]
                if sim is not None:
                    ly += 18
                    self._t(canvas, f"similarity to upstream (loci elf diff): {sim * 100:.0f} %", cx, ly, 0.4, MUTED)
            loops_bottom = max(loops_bottom, ly)

        # side-by-side listing
        ly0 = max(y0 + 330, loops_bottom + 44)
        lh = 17
        nrows = (y0 + h - 34 - ly0) // lh
        self._t(canvas, "ASSEMBLY  (before | after)", x, ly0 - 10, 0.45, MUTED)
        if same:
            rows = [(l, l, "equal") for l in before["lines"]]
            start = 0
        else:
            rows = self.rows[key]
            start = self.scroll[key]
            self._t(canvas, f"rows {start + 1}-{min(start + nrows, len(rows))} of {len(rows)}   "
                    "[ ] scroll   N / P next / previous change", x + colw + 20, ly0 - 10, 0.4, MUTED)
        for i, (l, r, tag) in enumerate(rows[start:start + nrows]):
            yy = ly0 + 6 + i * lh
            for side, item in enumerate((l, r)):
                cx = x + side * (colw + 20)
                if item is None:
                    continue
                if tag != "equal":
                    bg = DEL_BG if side == 0 else INS_BG
                    cv2.rectangle(canvas, (cx - 4, yy), (cx + colw, yy + lh - 1), bg, -1)
                if item["kind"] == "block":
                    red = side == 1 and any(f"loop-header {l['id']}," in item["text"] for l in nested_rows)
                    self._m(canvas, item["text"][:62], cx, yy + 13, RED if red else AMBER)
                else:
                    self._m(canvas, "    " + item["text"][:58], cx, yy + 13, TEXT if tag != "equal" else MUTED)
        self._t(canvas, "B back to video", x, y0 + h - 12, 0.4, MUTED)

    @staticmethod
    def _t(img, s, x, y, scale, color, thick=1):
        cv2.putText(img, s, (int(x), int(y)), FONT, scale, color, thick, cv2.LINE_AA)

    @staticmethod
    def _m(img, s, x, y, color):
        cv2.putText(img, s, (int(x), int(y)), MONO, 1.0, color, 1, cv2.LINE_AA)
