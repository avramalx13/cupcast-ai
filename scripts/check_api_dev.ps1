param(
    [string]$ApiBase = "http://127.0.0.1:8000"
)

$ErrorActionPreference = "Stop"
$HealthUrl = "$($ApiBase.TrimEnd('/'))/health"

try {
    $Response = Invoke-WebRequest -UseBasicParsing -Uri $HealthUrl -TimeoutSec 5
    Write-Output "CupCast API OK: $HealthUrl returned $($Response.StatusCode)"
    exit 0
} catch {
    Write-Error "CupCast API is not reachable at $HealthUrl. Start it with: powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_api_dev.ps1"
    exit 1
}
