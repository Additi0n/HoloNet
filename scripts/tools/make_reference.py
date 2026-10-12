# -*- coding: utf-8 -*-
"""Patch pandoc's default reference.docx with Chinese-paper typography.

Body   : Times New Roman + 宋体, 10.5pt (五号), 1.5 line spacing, first-line indent
Heading: Times New Roman + 黑体
"""
import re
import shutil
import sys
import zipfile
from pathlib import Path

SRC = Path(sys.argv[1])
DST = Path(sys.argv[2])

BODY_FONTS = (
    '<w:rFonts w:ascii="Times New Roman" w:eastAsia="SimSun" '
    'w:hAnsi="Times New Roman" w:cs="Times New Roman" />'
)
HEAD_FONTS = (
    '<w:rFonts w:ascii="Times New Roman" w:eastAsia="SimHei" '
    'w:hAnsi="Times New Roman" w:cs="Times New Roman" />'
)

zin = zipfile.ZipFile(SRC)
names = zin.namelist()
data = {n: zin.read(n) for n in names}
zin.close()

styles = data["word/styles.xml"].decode("utf-8")

# --- document defaults: fonts + size + 1.5 line spacing --------------------
styles = styles.replace(
    '<w:rFonts w:asciiTheme="minorHAnsi" w:eastAsiaTheme="minorEastAsia" '
    'w:hAnsiTheme="minorHAnsi" w:cstheme="minorBidi" />',
    BODY_FONTS,
)
styles = styles.replace(
    '<w:sz w:val="24" />\n        <w:szCs w:val="24" />',
    '<w:sz w:val="21" />\n        <w:szCs w:val="21" />',
)
styles = styles.replace(
    '<w:pPr>\n        <w:spacing w:after="200" />\n      </w:pPr>',
    '<w:pPr>\n        <w:spacing w:after="120" w:line="360" w:lineRule="auto" />'
    "\n      </w:pPr>",
)

# --- headings and title use a bold sans CJK face ---------------------------
styles = re.sub(
    r'<w:rFonts w:asciiTheme="majorHAnsi"\s+w:eastAsiaTheme="majorEastAsia"'
    r'\s+w:hAnsiTheme="majorHAnsi"\s+w:cstheme="majorBidi" />',
    HEAD_FONTS,
    styles,
)

# --- body text: first-line indent of two CJK characters --------------------
styles = styles.replace(
    '<w:style w:type="paragraph" w:styleId="BodyText">\n'
    '    <w:name w:val="Body Text" />\n'
    '    <w:basedOn w:val="Normal" />\n'
    '    <w:link w:val="BodyTextChar" />\n'
    "    <w:qFormat />\n"
    "    <w:pPr>\n"
    '      <w:spacing w:before="180" w:after="180" />\n'
    "    </w:pPr>",
    '<w:style w:type="paragraph" w:styleId="BodyText">\n'
    '    <w:name w:val="Body Text" />\n'
    '    <w:basedOn w:val="Normal" />\n'
    '    <w:link w:val="BodyTextChar" />\n'
    "    <w:qFormat />\n"
    "    <w:pPr>\n"
    '      <w:spacing w:before="60" w:after="60" w:line="360" w:lineRule="auto" />\n'
    '      <w:ind w:firstLine="420" />\n'
    "    </w:pPr>",
)

data["word/styles.xml"] = styles.encode("utf-8")

# --- theme fonts as a fallback for anything still theme-bound --------------
if "word/theme/theme1.xml" in data:
    th = data["word/theme/theme1.xml"].decode("utf-8")
    th = th.replace('typeface="Calibri Light"', 'typeface="Times New Roman"')
    th = th.replace('typeface="Calibri"', 'typeface="Times New Roman"')
    data["word/theme/theme1.xml"] = th.encode("utf-8")

with zipfile.ZipFile(DST, "w", zipfile.ZIP_DEFLATED) as zout:
    for n in names:
        zout.writestr(n, data[n])

print("reference doc written:", DST)
