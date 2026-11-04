# sync_results.ps1 - copy benchmark artefacts from the Kaggle notebook
# output directories into paper/json and paper/images.
$ErrorActionPreference = 'Stop'

$Paper = $PSScriptRoot
$Kaggle = Join-Path (Split-Path $Paper -Parent) 'notebooks\kaggle'

$JsonDir = Join-Path $Paper 'json'
$FigDir = Join-Path $Paper 'images'
New-Item -ItemType Directory -Force -Path $JsonDir, $FigDir | Out-Null

$jobs = @(
    @{ Name = 'fasterrcnn'; Dir = '03_fasterrcnn\output' },
    @{ Name = 'yolov8s';    Dir = '04_yolov8s\output' },
    @{ Name = 'rtdetr';     Dir = '05_rtdetr\output' }
)

$copied = 0
foreach ($j in $jobs) {
    $src = Join-Path $Kaggle $j.Dir
    if (-not (Test-Path $src)) {
        Write-Host ("skip  {0,-11} (no output dir: {1})" -f $j.Name, $j.Dir)
        continue
    }

    # JSON result files
    foreach ($f in Get-ChildItem -Path $src -Filter '*.json' -File -ErrorAction SilentlyContinue) {
        if ($f.Name -eq 'val_gt.json' -or $f.Name -like 'preds_*') { continue }
        Copy-Item $f.FullName (Join-Path $JsonDir $f.Name) -Force
        Write-Host ("json  {0}" -f $f.Name)
        $copied++
    }

    # figures produced by the notebook
    foreach ($f in Get-ChildItem -Path $src -Include '*.png', '*.pdf' -File -Recurse -ErrorAction SilentlyContinue) {
        Copy-Item $f.FullName (Join-Path $FigDir $f.Name) -Force
        Write-Host ("fig   {0}" -f $f.Name)
        $copied++
    }
}

Write-Host ("`nDone. {0} file(s) copied into paper/json and paper/images." -f $copied)