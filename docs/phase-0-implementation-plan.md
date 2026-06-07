# Phase 0 — Mathematical Foundation — Implementation Plan

**Depends on:** Nothing  
**Blocks:** Phase I (LENS)  
**Languages:** C (kernels), C++ (types, autodiff), Python (smoke tests)

---

## 1. Objective

Establish the **axiomatic bedrock** before any Aurelis layer code:

- **A.** O(n) state-space recurrence infrastructure (associative scan)
- **B.** Spectral stability enforcement (bounded eigenvalues / contraction)
- **C.** Four-channel state partition layout (c | e | r | m)

Phase 0 delivers **no trainable model** — only primitives that Phase I composes.

---

## 2. Spec Mapping (from project.md)

| Axiom | Implementation artifact |
|-------|-------------------------|
| O(n) state-space | `scan_blelloch.c`, `recurrence_diag.c` |
| Lyapunov-spectral | `spectral_clamp()` in FWSE precursor |
| Self-reference partition | `StatePartition` struct in C++ |

---

## 3. Components to Build

### 3.1 Build System
- **CMake 3.20+** project `aurelis`
- Targets: `aurelis_c` (static), `aurelis_core` (static), `aurelis_py` (module)
- Link OpenBLAS (`-DAURELIS_USE_BLAS`)
- Optional: `-DAURELIS_USE_CUDA` for GPU scan (later)

### 3.2 Tensor & Memory (`src/core/tensor.cpp`)
```cpp
struct Tensor {
  float* data;
  int64_t* shape;
  int ndim;
  bool requires_grad;
};
```
- Row-major, float32 default
- `view`, `slice`, `cat` along channel dim
- Aligned alloc (64-byte) for SIMD

### 3.3 State Partition (`include/aurelis/state_partition.hpp`)
```cpp
struct StatePartition {
  int D, Dc, De, Dr, Dm;
  void split(const float* h, float* c, float* e, float* r, float* m);
  void merge(float* h, const float* c, const float* e, const float* r, const float* m);
};
```
Ratios: 0.5 / 0.2 / 0.2 / 0.1

### 3.4 Associative Scan — C (`src/c/scan_blelloch.c`)

**Operator per dim:** `(α₂,β₂) ⊕ (α₁,β₁) = (α₂α₁, α₂β₁+β₂)`

| Function | Purpose |
|----------|---------|
| `scan_forward(a, b, h0, out, n, D)` | Blelloch up-down per dimension |
| `scan_backward(...)` | Adjoint reverse scan |
| `scan_op_assoc_test()` | Unit test associativity |

**Complexity contract:** document and benchmark `work = O(nD)`, `depth = O(log n)` for parallel build.

### 3.5 Diagonal Recurrence (`src/c/recurrence_diag.c`)
- Element-wise: `h_t = a_t * h_{t-1} + b_t`
- Bridge to scan tuples `(a_t, b_t)`

### 3.6 Spectral Clamp (`src/core/spectral.cpp`)
```cpp
void clamp_forget_gate(float* a, int D, float eps_min);
// a_i <- (1 - eps_k) * sigmoid(a_i), per scale group
```

### 3.7 Minimal Autodiff (`src/core/autodiff.cpp`)
- Tape-based reverse mode over:
  - element-wise ops
  - matmul (BLAS)
  - scan (custom backward)
- No full graph optimizer yet — sufficient for 1-layer test

### 3.8 BLAS Matmul Wrapper (`src/c/matmul_blas.c`)
- `sgemm` wrapper: `C = alpha*A*B + beta*C`
- Used by CCM and future FWSE

### 3.9 Python Bindings (`python/aurelis/_core.cpp` via pybind11)
- Expose: `Tensor`, `scan_forward`, `StatePartition`
- Smoke test: `tests/python/test_phase0_scan.py`

---

## 4. Implementation Steps

| Step | Task | Owner tier | Done when |
|------|------|------------|-----------|
| 0.1 | CMake skeleton + fetch OpenBLAS | CMake | `cmake --build` succeeds |
| 0.2 | Tensor alloc/view | C++ | unit test passes |
| 0.3 | `scan_op` associativity test | C | 1000 random triples |
| 0.4 | Blelloch forward vs sequential | C | max error < 1e-5 |
| 0.5 | Blelloch backward vs finite diff | C | grad error < 1e-3 |
| 0.6 | StatePartition split/merge | C++ | round-trip exact |
| 0.7 | Autodiff through matmul | C++ | matches finite diff |
| 0.8 | pybind11 module | Python | `import aurelis` works |
| 0.9 | Benchmark scan vs n | Python | slope linear on log-log |

---

## 5. File Checklist

```
src/c/scan_blelloch.c
src/c/scan_blelloch.h
src/c/recurrence_diag.c
src/c/matmul_blas.c
src/core/tensor.cpp
src/core/autodiff.cpp
src/core/spectral.cpp
include/aurelis/state_partition.hpp
python/aurelis/_core.cpp
tests/c/test_scan.c
tests/cpp/test_tensor.cpp
tests/python/test_phase0_scan.py
CMakeLists.txt
```

---

## 6. Exit Criteria

- [ ] Associativity proven by test (matches project.md proof)
- [ ] Forward scan matches sequential recurrence for random (a,b), n ∈ {1,7,128,1024}
- [ ] Backward scan validated
- [ ] State partition indices match D_c/D_e/D_r/D_m spec
- [ ] O(n) benchmark: doubling n doubles time (±10%)

---

## 7. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Scan backward bugs | Compare to PyTorch-free manual grad on tiny n |
| BLAS not on Windows | Bundle OpenBLAS prebuilt or vcpkg |
| Autodiff scope creep | Phase 0 only supports ops needed for Phase I |

**Estimated duration:** 3–4 weeks
