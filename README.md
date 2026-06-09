# Project Aurelis

Next-generation O(n) generative substrate — custom C/C++/Python stack (no ML framework).

## Documentation

- [Architecture spec](project.md)
- [Master implementation plan](docs/implementation-plan.md)
- [Phase 0 plan](docs/phase-0-implementation-plan.md)
- [Phase 1 plan](docs/phase-1-implementation-plan.md)
- [Phase 2 plan](docs/phase-2-implementation-plan.md)
- [Phase 3 plan](docs/phase-3-implementation-plan.md)
- [Phase 4 plan](docs/phase-4-implementation-plan.md)
- [Phase 5 plan](docs/phase-5-implementation-plan.md)
- [Productionization plan](docs/productionization-plan.md)
- [Release checklist](docs/release-checklist.md)

## Build (Phases 0-5)

```powershell
cmake -S . -B build -DAURELIS_BUILD_TESTS=ON
cmake --build build --config Release
ctest --test-dir build -C Release
```

For a repeatable release validation pass, run:

```powershell
powershell -File .\scripts\verify_release.ps1
```

To generate a release package artifact from the validated build tree, run:

```powershell
cmake --build build --config Release --parallel
cpack --config build/CPackConfig.cmake -G ZIP
```

For a quick release benchmark and timing baseline, run:

```powershell
powershell -File .\tools\benchmark_release.ps1
```

The release scripts resolve CMake/CTest from PATH first, then fall back to standard install locations. Pass `-Generator "Ninja"` (or your CI generator) and `-Configuration Release` when you need to override the default toolchain in CI.

Optional BLAS (faster matmul): install OpenBLAS and ensure CMake `find_package(BLAS)` succeeds.

## Layout

```
include/aurelis/   Public headers
src/c/             C kernels (scan, matmul)
src/core/          C++ tensor, autodiff, state partition, logging, errors, profiler, config
src/lens/          Phase I - LENS (Linear Epistemic State Network)
src/aureum/        Phase II - AUREUM (Golden Module: uncertainty, manifold)
src/arc/           Phase III - ARC (Attractor-based Reasoning Core)
src/tesserae/      Phase IV - TESSERAE (Compression & Edge Synthesis)
src/convergence/   Phase V - CONVERGENCE (AGI Component Interface)
tests/             C and C++ unit tests
configs/           Configuration files
python/            Python bindings + examples
```

## All Phases Status

- [x] Phase 0 - Mathematical Foundation
- [x] Phase I - LENS
- [x] Phase II - AUREUM
- [x] Phase III - ARC
- [x] Phase IV - TESSERAE
- [x] Phase V - CONVERGENCE
- [x] Checkpointing: Save/Load model weights
- [x] Configuration: JSON-based config system
- [x] Logging: Structured logging
- [x] Error Handling: Robust, human-readable error messages
- [x] Python/C API: Bindings for core types
- [x] Profiling: Simple profiling system
- [ ] Documentation
- [ ] More Tests

## Run Training Demo (LENS)

```powershell
.\build\train_lens.exe
```

## Next Steps

- Add more comprehensive documentation
- Add more unit and integration tests
- Add acceleration (SIMD, GPU, TPU)
- Optimize performance-critical paths
- Add quantization support

## Elite Branch

This branch (`elite`) contains the full productionization features (checkpointing, config, logging, errors, profiling, Python bindings). The main branch (`main`) is kept as a clean, stable release with all core phases.
