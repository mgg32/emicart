Param(
    [string]$PythonExe = "py"
)

$ErrorActionPreference = "Stop"

# Install EmiCart on Windows.
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\scripts\install_windows.ps1
# Optional Python selector:
#   powershell -ExecutionPolicy Bypass -File .\scripts\install_windows.ps1 -PythonExe "py -3.11"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

Write-Host "Creating virtual environment..."
& $PythonExe -m venv .venv

$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    throw "Virtual environment python not found at $venvPython"
}

Write-Host "Upgrading pip..."
& $venvPython -m pip install --upgrade pip

Write-Host "Installing runtime dependencies..."
& $venvPython -m pip install numpy matplotlib scipy pyvisa pyvisa-py psutil zeroconf

Write-Host "Installing EmiCart package..."
& $venvPython -m pip install .

# Create desktop shortcut.
Write-Host "Creating desktop shortcut..."
$desktopPath = [Environment]::GetFolderPath("Desktop")
if ([string]::IsNullOrWhiteSpace($desktopPath) -or -not (Test-Path $desktopPath)) {
    Write-Warning "Desktop path not found. Skipping shortcut creation."
} else {
    $shortcutPath = Join-Path $desktopPath "EmiCart.lnk"
    $wsh = New-Object -ComObject WScript.Shell
    $shortcut = $wsh.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = $venvPython
    $shortcut.Arguments = "-m emicart"
    $shortcut.WorkingDirectory = $repoRoot
    $shortcut.Description = "Launch EmiCart"
    $shortcut.Save()
    Write-Host "Created shortcut: $shortcutPath"
}

Write-Host ""
Write-Host "Install complete."
Write-Host "Activate env: .\.venv\Scripts\Activate.ps1"
Write-Host "Run GUI:      python -m emicart"
