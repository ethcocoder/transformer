# Project Aurelis

Next-generation O(n) generative substrate — custom C/C++/Python stack (no ML framework).

## Documentation

- [Architecture spec](project.md)
- [Master implementation plan](docs/implementation-plan.md)
- [Phase 0 plan](docs/phase-0-implementation-plan.md)

## Build (Phase 0)

```powershell
cmake -S . -B build -DAURELIS_BUILD_TESTS=ON
cmake --build build --config Release
ctest --test-dir build -C Release
```

Optional BLAS (faster matmul): install OpenBLAS and ensure CMake `find_package(BLAS)` succeeds.

## Layout

```
include/aurelis/   Public headers
src/c/             C kernels (scan, matmul)
src/core/          C++ tensor, autodiff, state partition
tests/             C and C++ unit tests
```

## Phase 0 Status

- [x] Blelloch associative scan (forward + backward)
- [x] Tensor + state partition (c|e|r|m)
- [x] BLAS matmul with naive fallback
- [x] Minimal autodiff (add, mul, matmul)
- [x] Spectral forget-gate clamp

## Phase I (LENS) Status

- [x] IETCF — embedding + temporal control field (orthogonal R)
- [x] FWSE — spectral forget/inject gates
- [x] CSC + PTK — parallel scan + cross-channel mix
- [x] MGP — manifold gauge projection
- [x] SPI — four-channel state partition
- [x] OSH — logits + skip connection
- [x] `LensStack` via `LensModel` (multi-layer residual)
- [x] Training demo: `build/train_lens.exe`
- [ ] Full backward through all layers (OSH + embedding train now)
- [ ] Python bindings (optional)

Run training demo:

```powershell
.\build\train_lens.exe
```

## Next: Phase II (AUREUM)

See [docs/phase-2-implementation-plan.md](docs/phase-2-implementation-plan.md).
