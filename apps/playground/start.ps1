param(
    [ValidateRange(1024, 65535)][int]$Port = 8765,
    [ValidateSet('127.0.0.1', '0.0.0.0')][string]$BindAddress = '127.0.0.1',
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
$pythonPath = Join-Path $projectRoot '.venv/Scripts/python.exe'
$serverPath = Join-Path $PSScriptRoot 'server.py'
$logDirectory = Join-Path $projectRoot '.cache/playground'
$baseUrl = "http://127.0.0.1:$Port"

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'The project Python environment is missing. See apps/playground/README.md.'
}

$existing = $null
try { $existing = Invoke-RestMethod -Uri "$baseUrl/api/status" -TimeoutSec 2 } catch {}
if ($null -ne $existing -and $existing.app -eq 'veyra-playground') {
    $existingBind = if ($existing.bind_host) { $existing.bind_host } else { '127.0.0.1' }
    if ($existingBind -ne $BindAddress) {
        throw "The server is listening on $existingBind. Run stop.ps1 before restarting with -BindAddress $BindAddress."
    }
    Write-Output "Playground already running at $baseUrl ($($existing.phase))."
    if (-not $NoBrowser) { Start-Process -FilePath $baseUrl }
    return
}

$listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($listener) { throw "Port $Port is already occupied. Use -Port with another local port." }
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
$process = Start-Process -FilePath $pythonPath -WorkingDirectory $projectRoot `
    -ArgumentList @('-X', 'utf8', ('"' + $serverPath + '"'), '--port', $Port, '--host', $BindAddress) `
    -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $logDirectory "server-$Port.stdout.log") `
    -RedirectStandardError (Join-Path $logDirectory "server-$Port.stderr.log")

$available = $false
for ($attempt = 0; $attempt -lt 40; $attempt++) {
    if ($process.HasExited) { throw "Playground exited. Read .cache/playground/server-$Port.stderr.log." }
    try {
        $status = Invoke-RestMethod -Uri "$baseUrl/api/status" -TimeoutSec 1
        if ($status.app -eq 'veyra-playground') { $available = $true; break }
    } catch {}
    Start-Sleep -Milliseconds 250
}
if (-not $available) { throw "The server is still starting. Inspect .cache/playground/server-$Port.stderr.log and open $baseUrl." }
Write-Output "Playground listening on ${BindAddress}:$Port (PID $($process.Id)). The model loads in the background."
if (-not $NoBrowser) { Start-Process -FilePath $baseUrl }
