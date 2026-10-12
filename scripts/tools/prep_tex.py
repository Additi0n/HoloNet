# -*- coding: utf-8 -*-
"""Preprocess the LaTeX paper so pandoc can render it faithfully into .docx."""
import re
import sys
from pathlib import Path

SRC = Path(sys.argv[1])
DST = Path(sys.argv[2])
# where rasterised copies of vector graphics are written
ASSETS = Path(sys.argv[3]) if len(sys.argv) > 3 else DST.parent / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

LANG = "en" if "--lang=en" in sys.argv else "zh"

SEARCH_DIRS = [SRC.parent, SRC.parent / "images", SRC.parent / "icon"]

tex = SRC.read_text(encoding="utf-8")

# ---------------------------------------------------------------- 1. tikz/pgfplots
# The only tikzpicture is the training-curve group plot; a pre-rendered PNG of the
# exact same figure lives in images/ (make_training_curves.py builds both locales
# from the same .dat tables the pgfplots block reads).
_curves = "training_curves_acc_f1_en.png" if LANG == "en" else "training_curves_acc_f1.png"
tikz_pat = re.compile(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", re.S)
n_tikz = len(tikz_pat.findall(tex))
tex = tikz_pat.sub(
    r"\\includegraphics[width=\\textwidth]{images/%s}" % _curves, tex
)

# ---------------------------------------------------------------- 2. PDF graphics -> PNG
# docx cannot embed PDF, so rasterise every PDF figure at print resolution and
# point \includegraphics at the PNG. Rendering the actual PDF keeps the docx in
# sync with the PDF build -- images/ holds several near-identical draft renders,
# and picking one by name silently ships the wrong diagram.
RASTER_DPI = 150  # 1500pt-wide figure -> ~3100px, ~530 dpi at final print size
n_pdf = 0


def resolve(name):
    cands = [name] if Path(name).is_absolute() else []
    cands += [d / name for d in SEARCH_DIRS]
    cands += [d / Path(name).name for d in SEARCH_DIRS]
    for c in cands:
        if Path(c).is_file():
            return Path(c)
    return None


def pdf_to_png(m):
    global n_pdf
    opts, name = m.group(1) or "", m.group(2)
    src = resolve(name)
    if src is None:
        print(f"  !! PDF figure not found: {name}", file=sys.stderr)
        return m.group(0)
    import pymupdf

    out = ASSETS / (src.stem + ".png")
    doc = pymupdf.open(src)
    doc[0].get_pixmap(dpi=RASTER_DPI).save(out)
    doc.close()
    n_pdf += 1
    return r"\includegraphics%s{%s}" % (opts, out.as_posix())


tex = re.sub(
    r"\\includegraphics(\[[^\]]*\])?\{([^}]*\.pdf)\}", pdf_to_png, tex
)

# ---------------------------------------------------------------- 3. subfigure
# pandoc does not know \subfigure, and its docx writer lays every block of a
# multi-block figure out as one table row. So keep only the images as blocks
# (giving a clean side-by-side panel row, as in the PDF) and fold the panel
# labels into the figure caption. Relative widths such as 0.32\textwidth are
# mis-read as inches by the docx writer, hence the absolute width.
CJK_IDX = "abcdefghij"
sub_pat = re.compile(
    r"\\subfigure\[([^\]]*)\]\{(\\includegraphics(?:\[[^\]]*\])?\{[^}]*\})\}"
)
n_sub = len(sub_pat.findall(tex))


LEGEND_LEAD = " Panels, in order: " if LANG == "en" else " 面板依次为："
LEGEND_SEP = "; " if LANG == "en" else "、"
LEGEND_END = "." if LANG == "en" else "。"


def rewrite_figure(fm):
    block = fm.group(0)
    panels = sub_pat.findall(block)
    if not panels:
        return block
    # the docx writer multiplies an image's width by its table-column fraction
    # (1/n here), so pre-multiply to land on the intended on-page size
    n = len(panels)
    width = "%.2fin" % min(1.85 * n, 5.6)
    images = []
    labels = []
    for i, (label, img) in enumerate(panels):
        img = re.sub(r"\[[^\]]*\]", "[width=%s]" % width, img, count=1)
        images.append(img)
        fmt = "(%s) %s" if LANG == "en" else "（%s）%s"
        labels.append(fmt % (CJK_IDX[i % 10], label))
    block = sub_pat.sub("", block)
    block = re.sub(r"\n[ \t]*\n[ \t]*\n+", "\n\n", block)
    # re-insert the images as separate blocks right after \centering
    joined = "\n\n" + "\n\n".join(images) + "\n\n"
    block = block.replace(r"\centering", r"\centering" + joined, 1)
    # append the panel legend to the caption
    block = re.sub(
        r"\\caption\{((?:[^{}]|\{[^{}]*\})*)\}",
        lambda cm: r"\caption{" + cm.group(1) + LEGEND_LEAD + LEGEND_SEP.join(labels) + LEGEND_END + "}",
        block,
        count=1,
    )
    return block


tex = re.sub(r"\\begin\{figure\}.*?\\end\{figure\}", rewrite_figure, tex, flags=re.S)

# ---------------------------------------------------------------- 4. starred floats
tex = tex.replace(r"\begin{table*}", r"\begin{table}").replace(
    r"\end{table*}", r"\end{table}"
)
tex = tex.replace(r"\begin{figure*}", r"\begin{figure}").replace(
    r"\end{figure*}", r"\end{figure}"
)

# ---------------------------------------------------------------- 5. booktabs rules
# \cmidrule leaks its span argument into the docx table as literal text.
n_cmid = len(re.findall(r"\\cmidrule", tex))
tex = re.sub(r"\\cmidrule(\([a-z]*\))?\{[0-9]+-[0-9]+\}", "", tex)

# ---------------------------------------------------------------- 6. citations -> [n]
order = re.findall(r"\\bibitem\{([^}]*)\}", tex)
key2num = {k: i + 1 for i, k in enumerate(order)}


def cite_repl(m):
    """Render like the `cite` package does: sorted, with runs of 3+ compressed."""
    nums = sorted({key2num[k.strip()] for k in m.group(1).split(",") if k.strip() in key2num})
    if len(nums) != len(m.group(1).split(",")):
        print(f"  !! unresolved citation key in {m.group(0)}", file=sys.stderr)
    parts, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        parts.append(f"{nums[i]}-{nums[j]}" if j - i >= 2 else
                     ",".join(str(n) for n in nums[i:j + 1]))
        i = j + 1
    return "[" + ",".join(parts) + "]"


n_cite = len(re.findall(r"\\cite\{([^}]*)\}", tex))
tex = re.sub(r"\\cite\{([^}]*)\}", cite_repl, tex)
# pandoc drops \bibitem's optional label, so put the number into the entry text
tex = re.sub(
    r"\\bibitem\{([^}]*)\}",
    lambda m: r"\bibitem{%s}[%s] " % (m.group(1), key2num.get(m.group(1), "?")),
    tex,
)

# ---------------------------------------------------------------- 7. bibliography heading
# thebibliography's heading comes from the document class, which pandoc ignores.
# Idempotent: newer drafts already carry their own heading.
_bib_at = tex.find(r"\begin{thebibliography}")
_bib_title = "References" if LANG == "en" else "参考文献"
# look for a real heading command, not the word in a stripped-out comment
_before = re.sub(r"(?m)^\s*%.*$", "", tex[max(0, _bib_at - 400) : _bib_at])
if _bib_at >= 0 and not re.search(r"\\section\*?\{[^}]*\}", _before):
    tex = tex.replace(
        r"\begin{thebibliography}",
        "\\section*{%s}\n\n\\begin{thebibliography}" % _bib_title,
        1,
    )

# ---------------------------------------------------------------- 8. packages pandoc chokes on
tex = tex.replace(r"\usepackage[inkscapelatex=false]{svg}", "")

DST.write_text(tex, encoding="utf-8")
print(f"tikz replaced : {n_tikz}")
print(f"pdf->png      : {n_pdf}")
print(f"subfigure flat: {n_sub}")
print(f"cmidrule strip: {n_cmid}")
print(f"citations     : {n_cite} over {len(order)} bib entries")
print(f"written       : {DST}")
