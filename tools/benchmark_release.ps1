param(
  [string]$BuildDir = "build",
  [string]$Generator = "",
  [string]$Configuration = "Release"
)

$ErrorActionPreference = "Stop"

function Resolve-Tool([string]$name) {
    $candidate = Get-Command $name -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Source
    if ($candidate) { return $candidate }

    $fallbacks = @()
    if ($IsWindows) {
        $programFiles = $env:ProgramFiles
        if ($programFiles) { $fallbacks += Join-Path $programFiles "CMake\bin\$name.exe" }
        $programFilesX86 = ${env:ProgramFiles(x86)}
        if ($programFilesX86) { $fallbacks += Join-Path $programFilesX86 "CMake\bin\$name.exe" }
    } else {
        $fallbacks += "/usr/local/bin/$name"
        $fallbacks += "/opt/homebrew/bin/$name"
        $fallbacks += "/usr/bin/$name"
    }

    foreach ($fallback in $fallbacks) {
        if (Test-Path $fallback) { return $fallback }
    }

    throw "Required tool '$name' was not found on PATH or in standard install locations."
}

$cmake = Resolve-Tool "cmake"
$ctest = Resolve-Tool "ctest"

function Invoke-Step([string]$Label, [scriptblock]$Action) {
    $start = Get-Date
    Write-Host "==> $Label"
    & $Action 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "$Label failed" }
    $elapsed = (Get-Date) - $start
    Write-Host ("   elapsed: {0:mm\:ss\.fff}" -f $elapsed)
    return $elapsed
}

$results = [ordered]@{
    Configure = 0
    Build = 0
    Test = 0
}

$configureArgs = @("-S", ".", "-B", $BuildDir, "-DAURELIS_BUILD_TESTS=ON", "-DCMAKE_BUILD_TYPE=$Configuration")
if ($Generator) { $configureArgs += @("-G", $Generator) }

$results.Configure = Invoke-Step "Configuring release build" {
    & $cmake @configureArgs
}

$results.Build = Invoke-Step "Building release artifacts" {
    & $cmake --build $BuildDir --config $Configuration --parallel
}

$results.Test = Invoke-Step "Running CTest" {
    & $ctest --test-dir $BuildDir -C $Configuration --output-on-failure
}

Write-Host ""
Write-Host "Benchmark summary"
foreach ($key in $results.Keys) {
    Write-Host ("  {0}: {1:mm\:ss\.fff}" -f $key, $results[$key])
}

$report = Join-Path $PSScriptRoot "benchmark_release.log"
$lines = @(
    "Aurelis release benchmark",
    ("Generated: " + (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")),
    ""
)
foreach ($key in $results.Keys) {
    $lines += ("{0}: {1:mm\:ss\.fff}" -f $key, $results[$key])
}
$lines | Set-Content -Path $report
Write-Host "Benchmark report written to $report"
