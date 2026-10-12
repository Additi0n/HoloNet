# -*- coding: utf-8 -*-
"""Localise pandoc's hard-coded English section labels inside the produced .docx."""
import sys
import zipfile
from pathlib import Path

SRC = Path(sys.argv[1])
DST = Path(sys.argv[2])

REPLACEMENTS = [
    ("<w:t xml:space=\"preserve\">Abstract</w:t>", "<w:t xml:space=\"preserve\">摘　要</w:t>"),
    ("<w:t>Abstract</w:t>", "<w:t>摘　要</w:t>"),
    ("<w:t xml:space=\"preserve\">Table of Contents</w:t>", "<w:t xml:space=\"preserve\">目　录</w:t>"),
    ("<w:t>Table of Contents</w:t>", "<w:t>目　录</w:t>"),
]

zin = zipfile.ZipFile(SRC)
names = zin.namelist()
data = {n: zin.read(n) for n in names}
zin.close()

hits = 0
for target in ("word/document.xml",):
    xml = data[target].decode("utf-8")
    for old, new in REPLACEMENTS:
        hits += xml.count(old)
        xml = xml.replace(old, new)
    data[target] = xml.encode("utf-8")

with zipfile.ZipFile(DST, "w", zipfile.ZIP_DEFLATED) as zout:
    for n in names:
        zout.writestr(n, data[n])

print(f"localised labels: {hits}")
print("written:", DST)
