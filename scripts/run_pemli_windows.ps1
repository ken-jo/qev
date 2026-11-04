param(
  [Parameter(Mandatory=$true)][string]$Records,
  [Parameter(Mandatory=$true)][string]$BaselineCheckpoint,
  [Parameter(Mandatory=$true)][string]$BackboneRevision,
  [string]$OutputRoot = 'runs/pemli-real',
  [int]$Epochs = 2
)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
function Run-Python {
  & $script:Python @args
  if ($LASTEXITCODE -ne 0) { throw "Python failed: $args" }
}
if (Test-Path $OutputRoot) { throw 'Use a new output folder; existing experiments are immutable.' }
if (-not (Test-Path $Records)) { throw 'Records JSONL is missing.' }
if (-not (Test-Path (Join-Path $BaselineCheckpoint 'manifest.json'))) { throw 'Local QEV-0.8B checkpoint manifest is missing.' }
if ($BackboneRevision -notmatch '^[0-9a-f]{40}$') { throw 'Provide an immutable HF backbone SHA.' }
if (-not (Test-Path '.venv-pemli/Scripts/python.exe')) {
  py -3.12 -m venv .venv-pemli
  if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 venv creation failed.' }
}
$script:Python = (Resolve-Path '.venv-pemli/Scripts/python.exe').Path
Run-Python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
Run-Python -m pip install transformers==5.22.1 accelerate safetensors==0.8.0 pydantic==2.14.2 pillow==13.0.0 numpy==2.5.4 fastapi==0.162.0 uvicorn==0.54.0
$env:PYTHONPATH = (Resolve-Path 'src').Path
Run-Python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name()); print(torch.cuda.mem_get_info())"
New-Item -ItemType Directory -Path $OutputRoot | Out-Null
Run-Python -m pip freeze | Out-File -Encoding utf8 (Join-Path $OutputRoot 'environment.txt')
Run-Python -m qev.pemli_real train --records $Records --output (Join-Path $OutputRoot 'checkpoint') --revision $BackboneRevision --epochs $Epochs --device cuda
Run-Python -m qev.pemli_real calibrate --checkpoint (Join-Path $OutputRoot 'checkpoint') --records $Records --output (Join-Path $OutputRoot 'calibrated') --device cuda
Run-Python scripts/evaluate_pemli_real.py --backend pemli --checkpoint (Join-Path $OutputRoot 'calibrated') --records $Records --device cuda --output (Join-Path $OutputRoot 'pemli-test.json')
# Separate process: the compact model is released before loading QEV-0.8B.
Run-Python scripts/evaluate_pemli_real.py --backend qev08 --checkpoint $BaselineCheckpoint --records $Records --device cuda --output (Join-Path $OutputRoot 'qev08-test.json')
Run-Python scripts/compare_pemli_reports.py --pemli (Join-Path $OutputRoot 'pemli-test.json') --qev08 (Join-Path $OutputRoot 'qev08-test.json') --output (Join-Path $OutputRoot 'comparison.json')
