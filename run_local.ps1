$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$pythonPath = Join-Path $projectRoot ".conda\python.exe"
if (-not (Test-Path $pythonPath)) {
    Write-Host "Missing local Python environment at .conda\python.exe" -ForegroundColor Red
    Write-Host "Create it first with:" -ForegroundColor Yellow
    Write-Host 'conda create --prefix ".conda" python=3.8 -y' -ForegroundColor Yellow
    Write-Host 'conda run --prefix ".conda" python -m pip install -r requirements.txt' -ForegroundColor Yellow
    exit 1
}

$primaryUrl = "http://127.0.0.1:8080"
$localIps = @()
try {
    $localIps = Get-NetIPAddress -AddressFamily IPv4 |
        Where-Object {
            $_.IPAddress -ne "127.0.0.1" -and
            $_.IPAddress -notlike "169.254.*" -and
            $_.PrefixOrigin -ne "WellKnown"
        } |
        Select-Object -ExpandProperty IPAddress -Unique
} catch {
    $localIps = @()
}

Write-Host ""
Write-Host "Starting Predictive Analysis Project UI..." -ForegroundColor Cyan
Write-Host "Local URL: $primaryUrl" -ForegroundColor Green

if ($localIps.Count -gt 0) {
    foreach ($ip in $localIps) {
        Write-Host "Same-network URL: http://$ip`:8080" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "Upload a CT image in the browser to test prediction." -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop the server." -ForegroundColor Yellow
Write-Host ""

& $pythonPath app.py
