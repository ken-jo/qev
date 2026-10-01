param([ValidateRange(1024, 65535)][int]$Port = 8765)
$ErrorActionPreference = 'Stop'
$baseUrl = "http://127.0.0.1:$Port"
$status = Invoke-RestMethod -Uri "$baseUrl/api/status" -TimeoutSec 3
if ($status.app -ne 'veyra-playground') { throw 'This address is not a Veyra playground.' }
Invoke-RestMethod -Method Post -Uri "$baseUrl/api/shutdown" `
    -Headers @{ 'X-Veyra-Playground' = '1' } -TimeoutSec 3 | Out-Null
Write-Output 'Shutdown requested. Active inference finishes before the model and temporary photos are released.'
