$svgPath = "article/images/framework_pipeline_mermaid_raw.svg"
$outPath = "article/images/framework_pipeline_svg.svg"
$svg = [System.IO.File]::ReadAllText($svgPath, [System.Text.Encoding]::UTF8)
$svg = $svg.Replace('脳', '×')
function Get-DataUri($path) {
    if (-not (Test-Path $path)) { return "" }
    $bytes = [System.IO.File]::ReadAllBytes($path)
    $base64 = [Convert]::ToBase64String($bytes)
    return "data:image/svg+xml;base64,$base64"
}
$map = @{
    "my-svg-flowchart-a_file-0"       = "article/icon/icon_binary_files.svg"
    "my-svg-flowchart-a_icon-15"      = "article/icon/icon_conv_refine.svg"
    "my-svg-flowchart-b_i1-53"        = "article/icon/icon_feature_maps.svg"
    "my-svg-flowchart-b_i2-59"        = "article/icon/icon_feature_fusion.svg"
    "my-svg-flowchart-b_i3-65"        = "article/icon/icon_patch_tokenization.svg"
    "my-svg-flowchart-c_attnIcon-97"  = "article/icon/icon_transformer_router.svg"
    "my-svg-flowchart-c_pos-105"       = "article/icon/icon_feature_fusion.svg"
    "my-svg-flowchart-c_caps-109"      = "article/icon/icon_capsule_stack.svg"
    "my-svg-flowchart-c_len-111"       = "article/icon/icon_confidence_bars.svg"
}
foreach ($id in $map.Keys) {
    $uri = Get-DataUri $map[$id]
    if ($uri -eq "") { continue }
    $pattern = "(?s)(<g [^>]*?id=""$id""[^>]*?>.*?)<image\s+([^>]*?)(\s*/)?>"
    $svg = [Regex]::Replace($svg, $pattern, {
        param($m)
        $prefix = $m.Groups[1].Value
        $attrs = $m.Groups[2].Value
        $suffix = $m.Groups[3].Value
        $attrs = [Regex]::Replace($attrs, '\b(xlink:)?href="[^"]*"', "").Trim()
        return "$($prefix)<image $attrs href=""$uri"" xlink:href=""$uri""$suffix>"
    })
}
[System.IO.File]::WriteAllText($outPath, $svg, [System.Text.Encoding]::UTF8)
$ctData = ([Regex]::Matches($svg, 'href="data:image/svg\+xml;base64,')).Count
$ctEmpty = ([Regex]::Matches($svg, 'href=""')).Count
$sz = (Get-Item $outPath).Length
Write-Host "Report: DataURI_Count=$ctData, Empty_Count=$ctEmpty, FileSize=$sz"
