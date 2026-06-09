# Release Validation Checklist

Use this checklist before declaring a release candidate.

## Build and packaging
- [ ] Configure succeeds with the intended generator and toolchain
- [ ] Release build completes without errors
- [ ] `cmake --install` produces an install tree

## Test and quality
- [ ] `ctest` passes for the current build tree
- [ ] Production smoke test passes
- [ ] No failing or skipped critical tests remain

## Documentation and operations
- [ ] README install/use instructions are current
- [ ] Release notes mention the current build and validation status
- [ ] Known limitations or TODO items are documented

## Release gate
- [ ] The validation script `scripts/verify_release.ps1` runs successfully
- [ ] The benchmark script `tools/benchmark_release.ps1` runs with the same resolved CMake/CTest toolchain used in CI
- [ ] `cpack` generates a release artifact from the validated build tree
