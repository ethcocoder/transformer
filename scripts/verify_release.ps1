param(
    [string]$BuildDir = "build",
    [string]$InstallDir = "_install",
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

$configureArgs = @("-S", ".", "-B", $BuildDir, "-DAURELIS_BUILD_TESTS=ON", "-DCMAKE_BUILD_TYPE=$Configuration")
if ($Generator) { $configureArgs += @("-G", $Generator) }

Write-Host "==> Configuring release build"
& $cmake @configureArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "==> Building release artifacts"
& $cmake --build $BuildDir --config $Configuration --parallel
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "==> Running CTest"
& $ctest --test-dir $BuildDir -C $Configuration --output-on-failure
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "==> Installing release tree"
& $cmake --install $BuildDir --config $Configuration --prefix $InstallDir
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Release verification completed successfully."
