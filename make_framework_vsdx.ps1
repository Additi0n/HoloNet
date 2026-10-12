param(
    [string]$VsdxPath = "D:\workspace\kan_transformer\hOlo\article\framework_holonet.vsdx",
    [string]$LabelsPath = "D:\workspace\kan_transformer\hOlo\article\framework_labels.json",
    [string]$PreviewPath = "D:\workspace\kan_transformer\hOlo\article\images\framework_holonet_preview.png",
    [double]$PageW = 19.0,
    [double]$PageH = 7.8,
    [double]$RefW = 1900.0,
    [double]$RefH = 780.0
)

$ErrorActionPreference = 'Stop'

# ---- Load Unicode/Chinese labels from UTF-8 JSON (keeps this .ps1 ASCII-only) ----
$json = [System.IO.File]::ReadAllText($LabelsPath, [System.Text.Encoding]::UTF8)
$L = $json | ConvertFrom-Json

$FONT = '"Microsoft YaHei"'
$MONO = '"Consolas"'

function VX([double]$x) { $script:PageW * $x / $script:RefW }
function VY([double]$y) { $script:PageH - ($script:PageH * $y / $script:RefH) }
function RGBF([int]$r, [int]$g, [int]$b) { "RGB($r,$g,$b)" }

$C = @{
    Ink     = RGBF 65 85 109
    Blue    = RGBF 79 120 168
    Red     = RGBF 200 70 70
    Purple  = RGBF 122 88 166
    Green   = RGBF 90 132 64
    Teal    = RGBF 20 132 150
    Orange  = RGBF 214 137 60
    Black   = RGBF 33 37 43
    Gray    = RGBF 120 128 140
    White   = RGBF 255 255 255
    PanelA  = RGBF 245 247 255
    PanelB  = RGBF 255 248 239
    PanelC  = RGBF 247 243 255
    BoxBlue = RGBF 225 234 247
    BoxOrg  = RGBF 252 238 220
    BoxPur  = RGBF 236 228 248
    BoxGrn  = RGBF 228 242 224
    BoxGray = RGBF 232 235 240
    EdgeBlue= RGBF 120 156 198
    EdgeOrg = RGBF 214 160 96
    EdgePur = RGBF 160 132 196
    EdgeGrn = RGBF 132 176 120
}

function Set-Cell($shape, [string]$cell, [string]$formula) {
    try { $shape.CellsU($cell).FormulaU = $formula } catch {}
}

function Style-Shape($shape, [string]$fill, [string]$line, [double]$linePt = 1.0, [int]$dash = 1, [double]$roundPx = 0) {
    if ($fill -eq 'none') { Set-Cell $shape 'FillPattern' '0' }
    else { Set-Cell $shape 'FillPattern' '1'; Set-Cell $shape 'FillForegnd' $fill }
    if ($line -eq 'none') { Set-Cell $shape 'LinePattern' '0' }
    else {
        Set-Cell $shape 'LinePattern' ([string]$dash)
        Set-Cell $shape 'LineColor' $line
        Set-Cell $shape 'LineWeight' "$linePt pt"
    }
    if ($roundPx -gt 0) {
        Set-Cell $shape 'Rounding' ((VX $roundPx).ToString([Globalization.CultureInfo]::InvariantCulture) + ' in')
    }
}

function Set-Text($shape, [string]$text, [double]$size = 10, [string]$color = $C.Black, [bool]$bold = $false, [bool]$italic = $false, [int]$align = 1, [string]$font = $FONT) {
    $shape.Text = $text
    Set-Cell $shape 'Char.Font' $font
    Set-Cell $shape 'Char.Size' "$size pt"
    Set-Cell $shape 'Char.Color' $color
    $style = 0
    if ($bold) { $style += 1 }
    if ($italic) { $style += 2 }
    Set-Cell $shape 'Char.Style' ([string]$style)
    Set-Cell $shape 'Para.HorzAlign' ([string]$align)
    Set-Cell $shape 'VerticalAlign' '1'
    foreach ($m in 'TxtMarginLeft','TxtMarginRight','TxtMarginTop','TxtMarginBottom') { Set-Cell $shape $m '1 pt' }
}

function RectTL([double]$x, [double]$y, [double]$w, [double]$h, [string]$text = '', [string]$fill = 'none', [string]$line = $C.Black, [double]$size = 10, [bool]$bold = $false, [double]$linePt = 1.0, [int]$dash = 1, [double]$roundPx = 8, [string]$tcolor = $C.Black, [string]$font = $FONT) {
    $s = $script:Page.DrawRectangle((VX $x), (VY ($y + $h)), (VX ($x + $w)), (VY $y))
    Style-Shape $s $fill $line $linePt $dash $roundPx
    if ($text -ne '') { Set-Text $s $text $size $tcolor $bold $false 1 $font }
    return $s
}

function TextTL([double]$x, [double]$y, [double]$w, [double]$h, [string]$text, [double]$size = 10, [string]$color = $C.Black, [bool]$bold = $false, [bool]$italic = $false, [int]$align = 1) {
    $s = $script:Page.DrawRectangle((VX $x), (VY ($y + $h)), (VX ($x + $w)), (VY $y))
    Style-Shape $s 'none' 'none' 0 1 0
    Set-Text $s $text $size $color $bold $italic $align $FONT
    return $s
}

function LineTL([double]$x1, [double]$y1, [double]$x2, [double]$y2, [string]$color = $C.Ink, [double]$linePt = 1.6, [bool]$arrowEnd = $true, [int]$dash = 1) {
    $s = $script:Page.DrawLine((VX $x1), (VY $y1), (VX $x2), (VY $y2))
    Set-Cell $s 'LineColor' $color
    Set-Cell $s 'LineWeight' "$linePt pt"
    Set-Cell $s 'LinePattern' ([string]$dash)
    Set-Cell $s 'LineCap' '0'
    if ($arrowEnd) { Set-Cell $s 'EndArrow' '4'; Set-Cell $s 'EndArrowSize' '1' }
    return $s
}

# image-like stack of offset squares (RGBA tensor motif)
function ImageStackTL([double]$x, [double]$y, [double]$w, [double]$h, [string]$fill, [string]$line, [int]$n = 3, [double]$off = 7) {
    $shapes = @()
    for ($i = $n - 1; $i -ge 0; $i--) {
        $s = RectTL ($x + $i * $off) ($y + $i * $off) $w $h '' $fill $line 8 $false 1.0 1 5
        $shapes += $s
    }
    return $shapes
}

# vertical stack of thin rounded capsules
function CapsuleStackTL([double]$x, [double]$y, [double]$w, [double]$hEach, [int]$n, [string]$fill, [string]$line) {
    for ($i = 0; $i -lt $n; $i++) {
        RectTL $x ($y + $i * ($hEach + 3)) $w $hEach '' $fill $line 6 $false 0.8 1 4 | Out-Null
    }
}

# classification bars motif
function BarsTL([double]$x, [double]$y, [double]$w, [double]$h, [string]$fill, [string]$line) {
    $heights = @(0.45, 0.95, 0.6, 0.3, 0.75)
    $bw = $w / ($heights.Count * 1.6)
    for ($i = 0; $i -lt $heights.Count; $i++) {
        $bh = $h * $heights[$i]
        $bx = $x + $i * ($bw * 1.6)
        RectTL $bx ($y + $h - $bh) $bw $bh '' $fill $line 6 $false 0.8 1 2 | Out-Null
    }
}

function Draw-Figure {
    # ---------- Panel containers ----------
    RectTL 24 44 590 648 '' $C.PanelA $C.EdgeBlue 1.4 1 18 | Out-Null
    RectTL 640 44 590 648 '' $C.PanelB $C.EdgeOrg 1.4 1 18 | Out-Null
    RectTL 1256 44 620 648 '' $C.PanelC $C.EdgePur 1.4 1 18 | Out-Null

    TextTL 24 700 590 40 $L.a_title 15 $C.Blue $true | Out-Null
    TextTL 640 700 590 40 $L.b_title 15 $C.Orange $true | Out-Null
    TextTL 1256 700 620 40 $L.c_title 15 $C.Purple $true | Out-Null

    # ===================== Panel A =====================
    RectTL 52 86 80 42 $L.exe $C.BoxGray $C.Gray 9 $false 1.0 1 5 | Out-Null
    RectTL 52 134 80 42 $L.elf $C.BoxGray $C.Gray 9 $false 1.0 1 5 | Out-Null
    RectTL 52 182 80 42 $L.apk $C.BoxGray $C.Gray 9 $false 1.0 1 5 | Out-Null
    TextTL 134 96 56 28 $L.bin_read 8.5 $C.Gray $false | Out-Null
    LineTL 134 155 188 155 $C.Ink 1.6 $true | Out-Null

    RectTL 192 92 150 126 $L.bytes $C.White $C.Gray 9 $false 1.0 1 4 $C.Black $MONO | Out-Null
    TextTL 192 222 150 20 $L.byte_label 8.5 $C.Gray $false | Out-Null
    TextTL 344 96 40 28 $L.bin2img 9 $C.Blue $true | Out-Null
    LineTL 344 155 392 155 $C.Ink 1.6 $true | Out-Null

    ImageStackTL 394 100 116 116 $C.BoxBlue $C.EdgeBlue 3 7 | Out-Null
    TextTL 392 228 130 20 $L.rgba 9 $C.Blue $true | Out-Null
    LineTL 456 232 456 270 $C.Ink 1.6 $true | Out-Null

    RectTL 318 274 270 46 $L.conv1 $C.BoxBlue $C.EdgeBlue 9.5 $false 1.0 1 6 | Out-Null
    TextTL 318 322 270 18 $L.conv1_sub 8.5 $C.Gray $false | Out-Null
    LineTL 453 342 453 366 $C.Ink 1.6 $true | Out-Null
    RectTL 318 368 270 46 $L.conv2 $C.BoxBlue $C.EdgeBlue 9.5 $false 1.0 1 6 | Out-Null
    TextTL 318 416 270 18 $L.conv2_sub 8.5 $C.Gray $false | Out-Null
    LineTL 453 436 453 462 $C.Ink 1.6 $true | Out-Null

    ImageStackTL 396 466 116 92 $C.BoxBlue $C.EdgeBlue 3 6 | Out-Null
    TextTL 396 564 130 20 $L.f2 10 $C.Blue $true | Out-Null

    RectTL 60 604 510 30 $L.legend $C.White $C.Gray 9 $false 0.8 1 6 $C.Gray | Out-Null

    # A -> B flow arrow
    LineTL 514 512 648 512 $C.Ink 2.0 $true | Out-Null

    # ===================== Panel B =====================
    TextTL 700 70 330 22 $L.b_in 9.5 $C.Orange $true | Out-Null
    RectTL 700 96 330 46 $L.dwconv $C.BoxOrg $C.EdgeOrg 9.5 $false 1.0 1 6 | Out-Null
    LineTL 865 142 865 168 $C.Ink 1.6 $true | Out-Null

    RectTL 690 172 350 76 '' $C.White $C.EdgeOrg 1.0 1 6 | Out-Null
    TextTL 690 178 350 24 $L.rbf_title 9.5 $C.Orange $true | Out-Null
    TextTL 690 206 350 30 $L.rbf_formula 9.5 $C.Black $false $true | Out-Null
    LineTL 865 248 865 274 $C.Ink 1.6 $true | Out-Null

    RectTL 700 278 330 46 $L.spline $C.BoxOrg $C.EdgeOrg 9.5 $false 1.0 1 6 | Out-Null
    LineTL 865 324 865 350 $C.Ink 1.6 $true | Out-Null

    RectTL 668 354 394 64 $L.kan $C.BoxGrn $C.EdgeGrn 9.5 $false 1.0 1 6 | Out-Null
    LineTL 865 418 865 444 $C.Ink 1.6 $true | Out-Null

    RectTL 690 448 350 54 $L.fusion $C.BoxOrg $C.EdgeOrg 10 $true 1.0 1 6 | Out-Null
    TextTL 690 508 350 22 $L.b_out 9 $C.Gray $false | Out-Null

    # shortcut (residual) dashed path along the left
    LineTL 660 119 660 386 $C.Green 1.4 $false 2 | Out-Null
    LineTL 660 386 668 386 $C.Green 1.4 $true 2 | Out-Null
    TextTL 590 244 70 16 $L.shortcut 7.5 $C.Green $false $true | Out-Null

    # B -> C flow arrow
    LineTL 1042 475 1256 475 $C.Ink 2.0 $true | Out-Null

    # ===================== Panel C =====================
    RectTL 1430 66 262 32 $L.ctx $C.BoxPur $C.EdgePur 9 $false 1.0 1 14 | Out-Null
    LineTL 1561 98 1561 116 $C.Ink 1.4 $true | Out-Null

    RectTL 1356 118 350 56 $L.pcaps $C.BoxPur $C.EdgePur 9.5 $false 1.0 1 6 | Out-Null
    CapsuleStackTL 1744 110 64 14 5 $C.BoxPur $C.EdgePur | Out-Null
    LineTL 1531 174 1531 198 $C.Ink 1.4 $true | Out-Null

    RectTL 1356 200 350 46 $L.lin_pe $C.BoxBlue $C.EdgeBlue 9.5 $false 1.0 1 6 | Out-Null
    LineTL 1531 246 1531 268 $C.Ink 1.4 $true | Out-Null

    RectTL 1376 270 310 40 $L.concat $C.BoxBlue $C.EdgeBlue 9.5 $false 1.0 1 6 | Out-Null
    LineTL 1531 310 1531 326 $C.Ink 1.4 $true | Out-Null

    # transformer encoder container
    RectTL 1336 328 380 168 '' $C.White $C.EdgePur 1.2 2 8 | Out-Null
    TextTL 1346 334 360 20 $L.tf_title 9.5 $C.Purple $true | Out-Null
    RectTL 1356 360 340 40 $L.mhsa $C.BoxOrg $C.EdgeOrg 9.5 $false 1.0 1 6 | Out-Null
    RectTL 1356 408 165 34 $L.addnorm $C.BoxGray $C.Gray 9 $false 1.0 1 6 | Out-Null
    RectTL 1531 408 165 34 $L.ffn $C.BoxGray $C.Gray 9 $false 1.0 1 6 | Out-Null
    RectTL 1356 450 340 34 $L.addnorm2 $C.BoxGray $C.Gray 9 $false 1.0 1 6 | Out-Null
    LineTL 1531 496 1531 520 $C.Ink 1.4 $true | Out-Null

    RectTL 1356 522 350 44 $L.vcaps $C.BoxPur $C.EdgePur 9.5 $false 1.0 1 6 | Out-Null
    LineTL 1531 566 1531 588 $C.Ink 1.4 $true | Out-Null

    RectTL 1356 590 350 44 $L.logits $C.BoxGrn $C.EdgeGrn 10 $true 1.0 1 6 | Out-Null
    BarsTL 1744 590 64 44 $C.BoxGrn $C.EdgeGrn | Out-Null
    TextTL 1716 636 120 20 $L.predict 9 $C.Green $true | Out-Null
}

# ---- backup if file already exists ----
if (Test-Path -LiteralPath $VsdxPath) {
    $backup = [IO.Path]::ChangeExtension($VsdxPath, $null) + ("backup-" + (Get-Date -Format 'yyyyMMdd-HHmmss') + ".vsdx")
    Copy-Item -LiteralPath $VsdxPath -Destination $backup
    Write-Output "Backup: $backup"
}

$visio = New-Object -ComObject Visio.InvisibleApp
$doc = $visio.Documents.Add("")
$script:Page = $doc.Pages.Item(1)
$script:PageW = $PageW
$script:PageH = $PageH
$script:RefW = $RefW
$script:RefH = $RefH

$script:Page.PageSheet.CellsU('PageWidth').FormulaU = "$PageW in"
$script:Page.PageSheet.CellsU('PageHeight').FormulaU = "$PageH in"

Draw-Figure

# save as .vsdx (format 0 = current Visio drawing)
$doc.SaveAs($VsdxPath)
if ($PreviewPath) {
    $script:Page.Export($PreviewPath)
    Write-Output "Preview: $PreviewPath"
}
$doc.Close()
$visio.Quit()
Write-Output "Saved: $VsdxPath  (shapes: drawn)"
