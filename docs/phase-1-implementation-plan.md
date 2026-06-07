# Phase I — LENS — Implementation Plan

**Depends on:** Phase 0  
**Blocks:** Phase II (AUREUM)  
**Spec reference:** project.md § Phase I — 7 components

---

## 1. Objective

Replace Transformer attention with a strictly **O(n)** sequence transducer:

- 7 components: IETCF, FWSE, CSC, MGP, SPI, PTK, OSH
- L-layer stack with conservative residual flow
- Train next-token prediction; e/r/m channels get L2 aux loss only

---

## 2. Component → Code Mapping

| Component | C++ module | C kernel |
|-----------|------------|----------|
| **1 IETCF** — embed + TCG | `src/lens/ietcf.cpp` | orthogonal `R` apply in `src/c/orthogonal.c` |
| **2 FWSE** — fast weights | `src/lens/fwse.cpp` | sigmoid/silu fused |
| **3 CSC** — DRC, CCM, MSSP | `src/lens/csc.cpp` | scan via `scan_blelloch.c` |
| **4 MGP** — gauge projection | `src/lens/mgp.cpp` | Cholesky solve in `src/c/linalg_small.c` |
| **5 SPI** — 4-channel split | `src/lens/spi.cpp` | — |
| **6 PTK** — parallel training | `src/lens/ptk.cpp` | wraps scan |
| **7 OSH** — logits | `src/lens/osh.cpp` | matmul BLAS |

---

## 3. Data Flow (one layer)

```
token x_t
  → IETCF → (x_t, γ_t)
  → FWSE(x_t, γ_t) → (a_t, b_t)
  → PTK/CSC scan → h_t^raw
  → CCM: h_t = M * h_t^raw + c_bias
  → MGP → h_t^out
  → SPI → (c_t, e_t, r_t, m_t)   [e,r,m: placeholder + L2]
  → residual: x^{(l+1)} = c_t + x^{(l)}
```

Final layer: **OSH(c_t)** → logits.

---

## 4. Implementation Steps

### Step 1 — IETCF (Week 1)
- `Embedding` table: `E[vocab, d_model]`
- TCG: learned orthogonal `R` via Cayley transform or Householder chain
- `τ_t = R @ τ_{t-1}`; `γ_t = W_γ @ τ_t + b_γ`
- **C:** `orthogonal_apply(R, tau_in, tau_out, d_tau)`

### Step 2 — FWSE (Week 1–2)
- Gate: `z = silu(W_gate @ x + W_ctrl @ γ + b)`
- Forget: per-scale `a_i = (1-ε_k) * σ(W_a @ z + b_a)`
- Inject: `b = σ(...) ⊙ silu(...)`
- Wire `spectral_clamp` from Phase 0

### Step 3 — CSC + PTK (Week 2–3)
- Pack `(a_t, b_t)` per timestep → scan
- `h0` from initial state or zeros
- CCM: one `D×D` matmul per token (BLAS)
- MSSP: index sets `I_k` for k=1..4 with fixed ε_k

### Step 4 — MGP (Week 3)
- Learned `μ`, Cholesky `L` (lower tri, softplus diag)
- Forward: whiten → normalize to sphere radius √D → add μ
- Backward: custom through normalize + triangular solve

### Step 5 — SPI (Week 3)
- Slice `c` from first D_c dims
- Linear maps `W_e, W_r, W_m` for placeholder channels
- Aux loss: `L_aux = ||e||² + ||r||² + ||m||²`

### Step 6 — OSH + Stack (Week 4)
- `LensLayer` class composing 1–5
- `LensStack`: L layers, residual `x^{(l)} = c^{(l)} + x^{(l-1)}`
- OSH on final `c_t`

### Step 7 — Training (Week 5–6)
- **Python:** `python/train/train_phase1.py`
- Loss: `L_pred + 0.01 * L_stab + λ_aux * L_aux`
- Optimizer: Adam-like in C++ (or simple SGD first)
- Data: character/token LM, small corpus

### Step 8 — Validation (Week 6–7)
- Complexity sweep: n ∈ {128,256,512,1024,2048}
- Stability: max ||h_t|| bounded over 10k steps
- Loss decreases on tiny vocab

---

## 5. Key Types

```cpp
// include/aurelis/lens/lens_layer.hpp
class LensLayer {
 public:
  Tensor forward(const Tensor& x_embed, const Tensor& gamma, LensState& state);
  void backward(const Tensor& grad_out);
 private:
  IETCF ietcf_;  // per-layer only gamma gen if shared
  FWSE fwse_;
  CSC csc_;
  MGP mgp_;
  SPI spi_;
};
```

---

## 6. Hyperparameters (Phase I defaults)

| Param | Value |
|-------|-------|
| D | 256 (dev) / 512 (prod path) |
| L | 4 (dev) / 24 (prod path) |
| d_ff | 4D |
| ε_k | 0.9, 0.5, 0.1, 0.01 |
| λ_stab | 0.01 |
| λ_aux | 0.001 |

---

## 7. Tests

| Test | File | Assert |
|------|------|--------|
| Scan = recurrence | `tests/c/test_lens_recurrence.c` | error < 1e-4 |
| ρ(A) ≤ 1-ε | `tests/cpp/test_fwse_spectral.cpp` | max abs a < 1 |
| MGP norm | `tests/cpp/test_mgp.cpp` | ||L⁻¹(h-μ)||² ≈ D |
| Full backward | `tests/cpp/test_lens_backward.cpp` | grad check |
| LM overfit | `tests/python/test_phase1_overfit.py` | 1 batch → ~0 CE |

---

## 8. Exit Criteria

- [ ] 7 components implemented and wired
- [ ] L-layer forward/backward end-to-end
- [ ] Training reduces CE on small corpus
- [ ] No O(n²) code paths (audit grep)
- [ ] SPI allocates e/r/m; aux loss active
- [ ] Theorem 1–3 from spec: complexity + stability empirically OK

**Estimated duration:** 6–8 weeks after Phase 0
