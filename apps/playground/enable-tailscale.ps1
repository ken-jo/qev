param(
    [ValidateRange(1024, 65535)][int]$Port = 8765,
    [ValidateRange(1, 65535)][int]$HttpsPort = 443,
    [switch]$PrepareOnly
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
$localDirectory = Join-Path $projectRoot '.cache/playground'
$configPath = Join-Path $localDirectory "tailscale-$Port.json"
$baseUrl = "http://127.0.0.1:$Port"
$tailscalePath = (Get-Command tailscale.exe -ErrorAction Stop).Source

$statusJson = & $tailscalePath status --json
if ($LASTEXITCODE -ne 0) { throw 'Cannot read Tailscale. Use a terminal allowed to access the Tailscale service.' }
$status = $statusJson | ConvertFrom-Json
if ($status.BackendState -ne 'Running') { throw 'Connect Tailscale before enabling the playground.' }
$dnsName = $status.Self.DNSName.TrimEnd('.').ToLowerInvariant()
if ($dnsName -notmatch '^[a-z0-9-]+\.[a-z0-9.-]+\.ts\.net$') { throw 'This device needs a Tailscale DNS name.' }
$owner = $status.User.PSObject.Properties[$status.Self.UserID.ToString()].Value
if (-not $owner.LoginName) { throw 'This device must have a Tailscale user identity.' }
$authority = if ($HttpsPort -eq 443) { $dnsName } else { "${dnsName}:$HttpsPort" }
$remoteUrl = "https://$authority"
$serveKey = "${dnsName}:$HttpsPort"

$serveJson = & $tailscalePath serve status --json
if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect existing Tailscale Serve settings.' }
$serve = $serveJson | ConvertFrom-Json
$existingTcp = $null
if ($serve.TCP) { $existingTcp = $serve.TCP.PSObject.Properties[$HttpsPort.ToString()] }
if ($null -ne $existingTcp) {
    $web = $null
    if ($serve.Web) { $web = $serve.Web.PSObject.Properties[$serveKey].Value }
    $handlers = @()
    if ($web.Handlers) { $handlers = @($web.Handlers.PSObject.Properties) }
    if (-not $existingTcp.Value.HTTPS -or $handlers.Count -ne 1 -or
        $handlers[0].Name -ne '/' -or $handlers[0].Value.Proxy -ne $baseUrl) {
        throw "Tailscale port $HttpsPort already serves something else. Select another -HttpsPort."
    }
}
if ($serve.AllowFunnel -and $serve.AllowFunnel.PSObject.Properties[$serveKey].Value) {
    throw 'This port has Funnel enabled. Select a different HTTPS port for private access.'
}

$existingApp = $null
try { $existingApp = Invoke-RestMethod -Uri "$baseUrl/api/status" -TimeoutSec 3 } catch {}
if ($existingApp -and ($existingApp.app -ne 'veyra-playground' -or $existingApp.busy)) {
    throw 'The local port is serving another application or an inference is active. Retry when idle.'
}

$peerIps = @($status.Self.TailscaleIPs)
foreach ($peer in $status.Peer.PSObject.Properties.Value) {
    if ($peer.UserID -eq $status.Self.UserID -and -not $peer.Tags) { $peerIps += $peer.TailscaleIPs }
}
$policy = [ordered]@{
    origin = $remoteUrl
    allowed_login = $owner.LoginName
    allowed_ips = @($peerIps | Sort-Object -Unique)
}
New-Item -ItemType Directory -Path $localDirectory -Force | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
[IO.File]::WriteAllText((Join-Path $localDirectory "tailscale-serve-before-$stamp.json"), ($serveJson -join "`n"))
[IO.File]::WriteAllText($configPath, ($policy | ConvertTo-Json -Depth 4))
Write-Output "Prepared $remoteUrl for the device owner's current Tailscale devices."
Write-Output "Private access policy saved locally: $configPath"
if ($PrepareOnly) { return }

if ($existingApp) {
    & (Join-Path $PSScriptRoot 'stop.ps1') -Port $Port
    $stopped = $false
    for ($attempt = 0; $attempt -lt 40; $attempt++) {
        try { Invoke-RestMethod -Uri "$baseUrl/api/status" -TimeoutSec 1 | Out-Null }
        catch { $stopped = $true; break }
        Start-Sleep -Milliseconds 250
    }
    if (-not $stopped) { throw 'The server is still stopping. Restart after shutdown completes.' }
}
& (Join-Path $PSScriptRoot 'start.ps1') -Port $Port -NoBrowser
& $tailscalePath serve --bg "--https=$HttpsPort" $baseUrl
if ($LASTEXITCODE -ne 0) { throw 'Serve could not be enabled. Review the Tailscale message above.' }
Write-Output "Open $remoteUrl from an allowed device connected to Tailscale."
