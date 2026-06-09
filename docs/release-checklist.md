# Release Checklist

This checklist captures the minimum production gate for Aurelis before a release is considered ready.

## Build and packaging
- [x] CMake config works on the supported platform
- [x] `cmake --build` succeeds in Release mode
- [x] `cmake --install` produces a usable install tree
- [x] The release validation script is available at `scripts/verify_release.ps1`
- [ ] `cpack` produces a release artifact from the validated build tree

## Validation
- [x] `ctest` passes on the release build
- [x] The production smoke test passes
- [ ] No critical TODO or placeholder paths remain in release-critical code

## Documentation
- [x] README build and usage instructions are current
- [ ] Release notes summarize what changed
- [x] Installation and troubleshooting guidance are present

## Operations
- [x] Logging and profiling output are understandable
- [x] Error handling covers invalid input and file failures
- [x] Performance baseline is documented for major paths
