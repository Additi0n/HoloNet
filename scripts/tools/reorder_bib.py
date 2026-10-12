# -*- coding: utf-8 -*-
"""Reorder thebibliography into first-citation order (顺序编码制).

LaTeX numbers \\bibitem entries by their position in thebibliography, so putting
them in order of first \\cite in the body is what makes the printed numbers run
1, 2, 3, ... down the page.

Usage:  python reorder_bib.py <paper.tex> [--check]
        --check  report the current state without writing
"""
import re
import shutil
import sys
from pathlib import Path

TEX = Path(sys.argv[1])
CHECK_ONLY = "--check" in sys.argv[2:]

src = TEX.read_text(encoding="utf-8")

bib_start = src.index(r"\begin{thebibliography}")
bib_open_end = src.index("\n", bib_start) + 1
bib_end = src.index(r"\end{thebibliography}")
body, bib = src[:bib_start], src[bib_open_end:bib_end]

# --- order of first citation in the body -----------------------------------
cited = []
for m in re.finditer(r"\\cite\{([^}]*)\}", body):
    for key in (k.strip() for k in m.group(1).split(",")):
        if key and key not in cited:
            cited.append(key)

# --- split the bibliography into per-entry blocks --------------------------
marks = [(m.start(), m.group(1)) for m in re.finditer(r"\\bibitem\{([^}]*)\}", bib)]
entries = {}
current = []
for i, (pos, key) in enumerate(marks):
    stop = marks[i + 1][0] if i + 1 < len(marks) else len(bib)
    entries[key] = bib[pos:stop].rstrip() + "\n"
    current.append(key)

missing = [k for k in cited if k not in entries]
uncited = [k for k in current if k not in cited]

print(f"entries        : {len(current)}")
print(f"cited keys     : {len(cited)}")
print(f"cited undefined: {missing or 'none'}")
print(f"defined uncited: {uncited or 'none'}")

if missing:
    sys.exit("refusing to reorder while citations are unresolved")

# uncited entries (if any) keep their relative order at the end
order = cited + uncited
moved = [(k, current.index(k) + 1, i + 1) for i, k in enumerate(order) if current.index(k) != i]
print(f"already ordered: {'yes' if not moved else 'no'}")
for key, old, new in moved:
    print(f"   {key:26s} #{old:2d} -> #{new:2d}")

if CHECK_ONLY or not moved:
    sys.exit(0)

new_bib = "\n" + "\n".join(entries[k] for k in order) + "\n"
out = src[:bib_open_end] + new_bib + src[bib_end:]

shutil.copyfile(TEX, TEX.with_suffix(".tex.bak"))
TEX.write_text(out, encoding="utf-8")
print(f"\nrewritten      : {TEX}  (backup: {TEX.with_suffix('.tex.bak').name})")
