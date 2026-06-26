param(
  [string]$BuildDir = "build",
  [string]$InstallDir = "_install",
  [string]$Generator = "",
  [string]$Configuration = "Release"
)

$ErrorActionPreference = "Stop"

try {
  & "$PSScriptRoot\..\scripts\verify_release.ps1" -BuildDir $BuildDir -InstallDir $InstallDir -Generator $Generator -Configuration $Configuration
  if ($LASTEXITCODE -ne 0) { throw "Release validation failed" }
}
catch {
  throw
}

Write-Host "==> Release validation succeeded"
