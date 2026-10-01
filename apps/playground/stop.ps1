param([ValidateRange(1024, 65535)][int]$Port = 8765)
$ErrorActionPreference = 'Stop'
$baseUrl = "http://127.0.0.1:$Port"

# Allow an existing legacy process to shut down cleanly during migration.
$legacy = $null
try { $legacy = Invoke-RestMethod -Uri "$baseUrl/api/status" -TimeoutSec 2 } catch {}
if ($null -ne $legacy -and $legacy.app -eq 'veyra-playground') {
    Invoke-RestMethod -Method Post -Uri "$baseUrl/api/shutdown" `
        -Headers @{ 'X-Veyra-Playground' = '1' } -TimeoutSec 3 | Out-Null
    Write-Output 'Legacy playground shutdown requested; active inference finishes first.'
    return
}

$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
$processRecord = Join-Path $projectRoot ".cache/playground/server-$Port.process.json"
if (-not (Test-Path -LiteralPath $processRecord)) {
    throw 'No managed QEV server record exists. Stop the terminal that launched qev playground.'
}
$record = Get-Content -LiteralPath $processRecord -Raw | ConvertFrom-Json
if ($record.app -ne 'qev-playground' -or $record.port -ne $Port -or $record.project_root -ne $projectRoot) {
    throw 'The server record does not identify this QEV checkout and port.'
}
$process = Get-Process -Id ([int]$record.pid) -ErrorAction SilentlyContinue
if (-not $process) {
    Write-Output 'The recorded QEV process has already stopped.'
    return
}
if ([string]$process.StartTime.ToUniversalTime().Ticks -ne $record.start_ticks) {
    throw 'The recorded process ID has been reused; no process was stopped.'
}
if (-not [string]::Equals($process.Path, $record.executable, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'The recorded process executable does not match; no process was stopped.'
}
# The Windows virtual-environment launcher can own a Python child. Stop only this
# recorded process tree, after checking its start time and executable.
& taskkill.exe /PID $record.pid /T /F | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'Windows could not stop the recorded QEV process tree.' }
Write-Output "QEV playground on port $Port stopped."
