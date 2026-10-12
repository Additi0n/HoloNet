<#
.SYNOPSIS
    Convert the Chinese LaTeX paper in article/ to a Word .docx via pandoc.

.DESCRIPTION
    LaTeX constructs Word cannot represent are rewritten first (see
    tools/prep_tex.py): the pgfplots training-curve figure is swapped for its
    pre-rendered PNG, the PDF pipeline figure for its PNG, \subfigure panels are
    flattened, \cite keys are numbered, and \cmidrule spans are dropped.
    Fonts come from a patched reference document (Times New Roman + 宋体/黑体).

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\tex_to_docx.ps1
#>
param(
    [string]$TexFile = "article\malware_classification_paper_chinese_v3.tex",
    [string]$OutFile = "article\malware_classification_paper_chinese_v3.docx",
    [ValidateSet("zh", "en")]
    [string]$Lang = "zh"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$tools = Join-Path $repo "scripts\tools"
$work = Join-Path $env:TEMP "tex2docx"

$pandoc = @(
    "$env:LOCALAPPDATA\Pandoc\pandoc.exe",
    "C:\Program Files\Pandoc\pandoc.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $pandoc) {
    $cmd = Get-Command pandoc -ErrorAction SilentlyContinue
    if ($cmd) { $pandoc = $cmd.Source }
}
if (-not $pandoc) { throw "pandoc not found. Install it: winget install --id JohnMacFarlane.Pandoc -e" }

New-Item -ItemType Directory -Force -Path $work | Out-Null
$tex = Join-Path $repo $TexFile
$out = Join-Path $repo $OutFile
$articleDir = Split-Path -Parent $tex

Write-Host "[1/4] rewriting LaTeX for docx"
python (Join-Path $tools "prep_tex.py") $tex (Join-Path $work "prepped.tex") (Join-Path $work "assets") "--lang=$Lang"

& $pandoc -o (Join-Path $work "reference_default.docx") --print-default-data-file reference.docx
if ($Lang -eq "zh") {
    Write-Host "[2/4] building Chinese reference document"
    python (Join-Path $tools "make_reference.py") (Join-Path $work "reference_default.docx") (Join-Path $work "reference_cn.docx")
    $refDoc = Join-Path $work "reference_cn.docx"
}
else {
    Write-Host "[2/4] using the stock reference document (English manuscript)"
    $refDoc = Join-Path $work "reference_default.docx"
}

Write-Host "[3/4] pandoc latex -> docx"
$refArg = "--reference-doc=" + $refDoc
Push-Location $articleDir
try {
    & $pandoc (Join-Path $work "prepped.tex") `
        -f latex -t docx `
        --resource-path=".;images;icon;data" `
        $refArg `
        --number-sections --toc --toc-depth=3 `
        -o (Join-Path $work "stage1.docx")
    if ($LASTEXITCODE -ne 0) { throw "pandoc failed with exit code $LASTEXITCODE" }
}
finally { Pop-Location }

if ($Lang -eq "zh") {
    Write-Host "[4/4] localising section labels"
    python (Join-Path $tools "postprocess_docx.py") (Join-Path $work "stage1.docx") $out
}
else {
    Write-Host "[4/4] keeping pandoc's English section labels"
    Copy-Item (Join-Path $work "stage1.docx") $out -Force
}

Write-Host ""
Write-Host "Done -> $out"
Get-Item $out | Select-Object Name, Length, LastWriteTime
