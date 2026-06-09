# Release Checklist

This checklist captures the minimum production gate for Aurelis before a release is considered ready.

## Build and packaging
- [ ] CMake config works on the supported platform
- [ ] `cmake --build` succeeds in Release mode
- [ ] `cmake --install` produces a usable install tree

## Validation
- [ ] `ctest` passes on the release build
- [ ] The production smoke test passes
- [ ] No critical TODO or placeholder paths remain in release-critical code

## Documentation
- [ ] README build and usage instructions are current
- [ ] Release notes summarize what changed
- [ ] Installation and troubleshooting guidance are present

## Operations
- [ ] Logging and profiling output are understandable
- [ ] Error handling covers invalid input and file failures
- [ ] Performance baseline is documented for major paths
