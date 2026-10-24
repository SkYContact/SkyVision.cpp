#!/usr/bin/env python3
"""gen_backend_table.py - generate the README capability matrix from the spec table.

Single source of truth: `vqa::specs_to_json()` (src/core/runtime/vqa_spec.cpp),
dumped by `vqa-cli --specs` (also GET /api/specs). This script turns that JSON
into the "backend x input x task x streaming" quick-pick matrix that sits above
the hand-written detailed backend table in README.md / README.en.md.

The generated block lives between two HTML markers inside the README; do not
edit it by hand, re-run this script instead:

    # refresh the matrix in both READMEs (specs come straight from the binary)
    scripts/gen_backend_table.py --cli build-specs/Release/vqa-cli.exe --readme README.md
    scripts/gen_backend_table.py --cli build-specs/Release/vqa-cli.exe --readme README.en.md --lang en

    # or from a JSON dump (vqa-cli --specs > specs.json, kept out of the repo)
    scripts/gen_backend_table.py --specs specs.json --readme README.md

    # CI consistency check: the embedded matrix must match the spec table and
    # every architecture tag in the detailed table must be a known spec label
    scripts/gen_backend_table.py --cli build-specs/Release/vqa-cli.exe --readme README.md --check
    scripts/gen_backend_table.py --cli build-specs/Release/vqa-cli.exe --readme README.en.md --lang en --check

--check exits non-zero on: a stale/drifted matrix block, or an architecture tag
in the detailed table that the spec table does not know (rename / typo rot).
Spec rows that have no detailed-table row only produce a warning (use --strict
to promote them to errors), because the detailed prose is hand-curated.
"""

import argparse
import json
import re
import subprocess
import sys

BEGIN = "<!-- AUTO-GENERATED capability matrix - scripts/gen_backend_table.py, do not edit by hand -->"
END = "<!-- /AUTO-GENERATED capability matrix -->"

INPUT_ZH = {"image": "图像", "video": "视频", "audio": "音频", "text": "文本"}

DETAIL_HEADING = {
    "zh": re.compile(r"^## 支持的后端\s*$", re.M),
    "en": re.compile(r"^## Supported backends\s*$", re.M),
}


def load_specs(args):
    if args.specs:
        with open(args.specs, encoding="utf-8") as f:
            return json.load(f)
    if args.cli:
        out = subprocess.run([args.cli, "--specs"], capture_output=True, text=True, check=True)
        return json.loads(out.stdout)
    sys.exit("need --specs <file> or --cli <vqa-cli path>")


def flags(spec, lang):
    zh = lang == "zh"
    out = []
    if spec["pure_vision"]:
        out.append("纯视觉" if zh else "vision-only")
    if spec["needs_vision_gguf"]:
        out.append("需 `*-vision.gguf`" if zh else "needs `*-vision.gguf`")
    if spec["speech_output"]:
        out.append("语音输出" if zh else "speech output")
    return "，".join(out) if out else "—"


def matrix(specs, lang):
    zh = lang == "zh"
    lines = []
    if zh:
        lines.append("| 架构标签 | 别名 | 输入 | 任务 | 流式 | 标记 |")
        lines.append("|---|---|---|---|---|---|")
    else:
        lines.append("| Architecture tag | Aliases | Input | Tasks | Streaming | Flags |")
        lines.append("|---|---|---|---|---|---|")
    for s in specs:
        inputs = " + ".join(INPUT_ZH[i] if zh else i for i in s["inputs"])
        tasks = " ".join(s["tasks"]) if s["tasks"] else "—"
        streaming = "✅" if "streaming" in s["modes"] else "—"
        aliases = " ".join("`%s`" % a for a in s["aliases"]) if s["aliases"] else "—"
        lines.append("| `%s` | %s | %s | %s | %s | %s |" % (
            s["arch"], aliases, inputs, tasks, streaming, flags(s, lang)))
    return "\n".join(lines)


def read_readme(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def split_block(text):
    """Return (before, current block or None, after) around the markers."""
    m = re.search(re.escape(BEGIN) + r"\n(.*?)\n" + re.escape(END), text, re.S)
    if not m:
        i = text.find(BEGIN)
        if i >= 0:
            sys.exit("found BEGIN marker without END marker")
        return None
    return m.group(1)


def detail_table_tags(text, lang):
    """Backticked architecture tags in the hand-written detailed table.

    The detailed table is the first markdown table after the backends heading
    (the generated matrix sits before it and has no backticked cells except
    aliases - excluded because the matrix block is stripped first).
    """
    stripped = re.sub(re.escape(BEGIN) + r"\n.*?\n" + re.escape(END), "", text, flags=re.S)
    m = DETAIL_HEADING[lang].search(stripped)
    if not m:
        return []
    rest = stripped[m.end():]
    tags = []
    for line in rest.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            if tags:
                break  # table already started and ended
            continue
        cells = line.strip("|").split("|")
        if len(cells) >= 2 and not set(cells[1].strip()) <= {"-", " "}:
            # header/separator rows have a prose first cell; tag column is cells[1]
            tags += re.findall(r"`([^`]+)`", cells[1])
    return tags


def known_labels(specs):
    out = set()
    for s in specs:
        out.add(s["arch"])
        out.update(s["aliases"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--specs", help="specs JSON file (vqa-cli --specs > specs.json)")
    ap.add_argument("--cli", help="path to vqa-cli; run `--specs` directly")
    ap.add_argument("--readme", help="README file to update or check")
    ap.add_argument("--lang", choices=["zh", "en"], default="zh")
    ap.add_argument("--check", action="store_true", help="verify the README block and tags instead of writing")
    ap.add_argument("--strict", action="store_true", help="fail on spec rows missing from the detailed table")
    args = ap.parse_args()

    specs = load_specs(args)
    want = matrix(specs, args.lang)

    if not args.readme:
        print(want)
        return

    text = read_readme(args.readme)
    if args.check:
        problems = []
        have = split_block(text)
        if have is None:
            problems.append("%s: no generated matrix block found (run without --check to insert)" % args.readme)
        elif have != want:
            problems.append("%s: embedded capability matrix is stale (regenerate with this script)" % args.readme)
        known = known_labels(specs)
        unknown = [t for t in detail_table_tags(text, args.lang) if t not in known]
        if unknown:
            problems.append("%s: detailed table mentions unknown architecture tags %s "
                            "(renamed or misspelled - fix the spec table or the README)" % (args.readme, unknown))
        missing = [s["arch"] for s in specs
                   if s["arch"] not in detail_table_tags(text, args.lang)
                   and not (set(s["aliases"]) & set(detail_table_tags(text, args.lang)))]
        if missing:
            msg = "%s: spec rows without a detailed-table row: %s" % (args.readme, ", ".join(missing))
            problems.append(msg if args.strict else "WARNING: " + msg)
        for p in problems:
            print(p, file=sys.stderr)
        hard = [p for p in problems if not p.startswith("WARNING")]
        if hard:
            sys.exit(1)
        print("%s: matrix and architecture tags consistent with %d specs" % (args.readme, len(specs)))
        return

    block = BEGIN + "\n" + want + "\n" + END
    if BEGIN in text:
        text = re.sub(re.escape(BEGIN) + r"\n.*?\n" + re.escape(END), lambda _: block, text, count=1, flags=re.S)
    else:
        m = DETAIL_HEADING[args.lang].search(text)
        if not m:
            sys.exit("%s: backends heading not found" % args.readme)
        text = text[:m.end()] + "\n\n" + block + "\n" + text[m.end():]
    with open(args.readme, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print("%s: matrix updated (%d specs)" % (args.readme, len(specs)))


if __name__ == "__main__":
    main()
