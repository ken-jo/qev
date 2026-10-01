param(
    [ValidateRange(1024, 65535)][int]$Port = 8765,
    [ValidateSet('127.0.0.1', '0.0.0.0')][string]$BindAddress = '127.0.0.1',
    [ValidateSet('auto', 'cpu', 'cuda')][string]$Device = 'auto',
    [string]$Checkpoint,
    [string]$CacheDir,
    [switch]$Offline,
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
$pythonPath = Join-Path $projectRoot '.venv/Scripts/python.exe'
$logDirectory = Join-Path $projectRoot '.cache/playground'
$baseUrl = "http://127.0.0.1:$Port"
$processRecord = Join-Path $logDirectory "server-$Port.process.json"

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'The project Python environment is missing. Run uv sync --frozen first.'
}

function Get-QevPageTitle {
    $response = Invoke-WebRequest -UseBasicParsing -Uri "$baseUrl/config" -TimeoutSec 2
    # Gradio's config includes empty JSON property names. Windows PowerShell's
    # ConvertFrom-Json cannot represent these; use the already required Python.
    $title = $response.Content | & $pythonPath -X utf8 -c 'import json,sys; print(json.load(sys.stdin).get(sys.argv[1],str()))' title
    if ($LASTEXITCODE -ne 0) { throw 'The server did not return a valid QEV configuration.' }
    return $title
}

$existing = $null
try { $existing = Get-QevPageTitle } catch {}
if ($existing -eq 'QEV') {
    if (Test-Path -LiteralPath $processRecord) {
        $record = Get-Content -LiteralPath $processRecord -Raw | ConvertFrom-Json
        if ($record.bind_host -ne $BindAddress) {
            throw "The server is listening on $($record.bind_host). Stop it before changing the bind address."
        }
    }
    Write-Output "English QEV playground already running at $baseUrl."
    if (-not $NoBrowser) { Start-Process -FilePath $baseUrl }
    return
}
$probe = [System.Net.Sockets.TcpClient]::new()
$occupied = $false
try { $occupied = $probe.ConnectAsync('127.0.0.1', $Port).Wait(500) -and $probe.Connected } catch {}
finally { $probe.Dispose() }
if ($occupied) {
    throw "Port $Port is occupied. Stop the earlier server before launching the English playground."
}

New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
$gradioTemp = Join-Path $logDirectory 'gradio'
New-Item -ItemType Directory -Path $gradioTemp -Force | Out-Null
$arguments = @('-X', 'utf8', '-m', 'qev', 'playground', '--port', $Port, '--host', $BindAddress, '--device', $Device)
if ($Checkpoint) { $arguments += @('--checkpoint', ('"' + (Resolve-Path -LiteralPath $Checkpoint).Path + '"')) }
if ($CacheDir) { $arguments += @('--cache-dir', ('"' + $CacheDir + '"')) }
if ($Offline) { $arguments += '--offline' }
$previousPythonPath = $env:PYTHONPATH
$previousGradioTemp = $env:GRADIO_TEMP_DIR
try {
    $env:PYTHONPATH = Join-Path $projectRoot 'src'
    $env:GRADIO_TEMP_DIR = $gradioTemp
    $process = Start-Process -FilePath $pythonPath -WorkingDirectory $projectRoot `
        -ArgumentList $arguments -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $logDirectory "server-$Port.stdout.log") `
        -RedirectStandardError (Join-Path $logDirectory "server-$Port.stderr.log")
} finally {
    $env:PYTHONPATH = $previousPythonPath
    $env:GRADIO_TEMP_DIR = $previousGradioTemp
}
@{
    app = 'qev-playground'; pid = $process.Id; port = $Port; bind_host = $BindAddress
    start_ticks = [string]$process.StartTime.ToUniversalTime().Ticks
    executable = $pythonPath; project_root = $projectRoot
} | ConvertTo-Json | Set-Content -LiteralPath $processRecord -Encoding utf8

$available = $false
for ($attempt = 0; $attempt -lt 60; $attempt++) {
    if ($process.HasExited) { throw "Playground exited. Read .cache/playground/server-$Port.stderr.log." }
    try {
        $pageTitle = Get-QevPageTitle
        if ($pageTitle -eq 'QEV') { $available = $true; break }
    } catch {}
    Start-Sleep -Milliseconds 250
}
if ($available) {
    Write-Output "English QEV playground ready at $baseUrl; listening on ${BindAddress}:$Port (PID $($process.Id))."
    if (-not $NoBrowser) { Start-Process -FilePath $baseUrl }
} else {
    Write-Output "QEV is preparing the model (PID $($process.Id)). Progress: .cache/playground/server-$Port.stderr.log. Open $baseUrl when ready."
}
