# Phase V — CONVERGENCE — Implementation Plan

**Depends on:** Phase IV TESSERAE checkpoint + ODFFB runtime  
**Blocks:** None (terminal integration phase)  
**Spec reference:** project.md § Phase V — 7 components

---

## 1. Objective

Transform Aurelis into a **convergence engine**:

- Emit **Epistemic Frames** F_t each step for downstream AGI modules
- AGI bus API for world / motor / planning curricula
- **Self-Convergence Engine:** latent z self-updates from epistemic error
- **GFIS:** geometric forgetting immunity on manifold charts
- **CAR:** alignment drift detection and recovery

Token inference remains **O(n)**; self-update amortized every N_self steps.

---

## 2. Component → Code Mapping

| Component | Module |
|-----------|--------|
| **1 EFE** — FA, FCE, CSP | `src/convergence/efe.cpp` |
| **2 ACTI** — WMCS, MCGE, HPSG | `src/convergence/acti.cpp` |
| **3 SCE** — ERB, MGC, LCSU, CSG | `src/convergence/sce.cpp` |
| **4 GFIS** — CA, MCE, KBE | `src/convergence/gfis.cpp` |
| **5 CAR** — VMP, EDD, ARO | `src/convergence/car.cpp` |
| **6 CVK-V** — DCM, CMCA, SKIC | `src/convergence/cvk_v.cpp` |
| **7 P5IL** — FOR, SIGI, ABI | `src/convergence/p5il.cpp` |

**Python:** mock downstream trainers, integration tests, bus monitor CLI.

---

## 3. Epistemic Frame Structure

```cpp
struct EpistemicFrame {
  Tensor c;      // D_c
  Tensor e;      // D_e
  Tensor d;      // D_m  domain on M
  Tensor alpha;  // k=4 reasoning mode
  float kappa;
  Tensor sigma;  // D_e
  float H_reason;
};
```

**FA:** assemble from active channels after 4-pass forward.  
**FCE:** MLP enc → `f_t ∈ R^{D_bus}` (D_bus ≪ D_F).  
**CSP:** `u_t^(τ) = W_csp^(τ) @ f_t + b` for τ ∈ {world, motor, plan}.

---

## 4. AGI Bus Interface (ABI)

```cpp
// include/aurelis/convergence/agi_bus.hpp
class AgiBus {
 public:
  Tensor read(int t, TaskType tau);
  void write(int t, TaskType tau, const Tensor& delta_down);
  void subscribe(TaskType tau, DownstreamCallback cb);
};
```

**Phase V MVP:** in-process mock modules (C++ classes), not distributed RPC.

| Mock module | Input | Loss |
|-------------|-------|------|
| WorldModelMock | u^(world) | MSE + Lipschitz grad penalty |
| MotorMock | u^(motor) | κ-weighted BC loss |
| PlanMock | u^(plan) | graph scaffold from HPSG |

---

## 5. Self-Convergence Engine (SCE)

### ERB
- Circular buffer B=10^4: `(x, y, F, L_epistemic)`
- **C++** ring buffer; Python fills during training

### MGC + LCSU
```
L_inner(z) = mean[ CE + λ_e |κ - y_acc| ]
z_new = Proj_Z(z_old - η_self * ∇_z L_inner)
```
- Weights via INR_ψ(z) — backprop through synthesis (Phase IV path)
- **SIGI:** compute grad on shadow copy z'; apply async after sequence batch

### CSG
- Accept update only if ΔL_inner > τ_improve
- Else decay η_self *= γ_decay

**Schedule:** every N_self = 2048 sequences (config).

---

## 6. GFIS + CAR

### Chart Allocator
- On new domain centroid μ_{K+1}: if min_k d_M(μ_{K+1}, μ_k) > δ_chart → new chart
- **Python:** log chart registry JSON

### MCE
- On self-update: `L_preserve` on frozen chart pairs
- Block metric updates on frozen chart support

### CAR
- EDD: EMA of |κ - y_acc|
- If ECE > τ_drift for T_drift steps → ARO: revert z_safe, restorative grad on ECE

---

## 7. Implementation Steps

### Step 1 — EFE + FOR (Week 1–2)
- Wire frame assembly post-forward
- Optional FCE training with variational bottleneck
- Route f_t to AgiBus each step

### Step 2 — ACTI mocks (Week 2–3)
- WMCS, MCGE, HPSG produce targets from batch metadata
- Train adapter heads W_csp^(τ) only (freeze channels 0–25% curriculum)

### Step 3 — SCE (Week 3–5)
- Implement ERB + shadow-z gradient
- LCSU with L2 ball projection R_z
- CSG gate + logging

### Step 4 — GFIS (Week 4–5)
- Chart table + freeze masks on G_m updates
- KBE: block-diagonal metric expansion

### Step 5 — CAR + CVK-V (Week 5–6)
- EDD / ARO pipeline
- SKIC rollback on failed checks (calib, κ monotonicity, metric distortion)
- DCM: monitor mock downstream loss decrease

### Step 6 — P5IL integration (Week 6–7)
- Full training curriculum per spec (4 stages)
- Stress test: adversarial domain injection

### Step 7 — End-to-end demo (Week 7–8)
- **Python:** `python/train/train_phase5.py`
- Aurelis trains WorldModelMock while self-updating z
- Report: downstream converges, SKIC pass rate, no frozen-chart distortion

---

## 8. Training Curriculum

| Phase | Steps | Action |
|-------|-------|--------|
| 1 | 0–25% | Freeze c,e,r,m; train EFE + ACTI adapters |
| 2 | 25–60% | SCE on with η_self=1e-5 |
| 3 | 60–85% | Sequential new domains + GFIS |
| 4 | 85–100% | CAR stress tests |

---

## 9. Tests

| Test | Assert |
|------|--------|
| Frame size | O(D) constant per step |
| SCE descent | L_inner decreases when CSG accepts |
| GFIS | frozen chart distances unchanged after new domain |
| SKIC rollback | bad update reverts z |
| Amortized cost | self-update adds < 5% wall time amortized |
| DCM | mock world loss ↓ over epochs |

---

## 10. Exit Criteria

- [ ] Epistemic frames exported each token step
- [ ] AgiBus mock modules receive u_t and converge
- [ ] Self-update improves κ calibration without SKIC failures
- [ ] New domain adds chart without frozen-chart geodesic change
- [ ] Full stack documented: Phase 0→V runnable from README

**Estimated duration:** 6–8 weeks after Phase IV

---

## 11. Project Completion

After Phase V, deliver:

1. `aurelis_infer` — edge inference (TESSERAE)
2. `aurelis_train` — Python entry per phase
3. `model.aurl` — example compressed checkpoint
4. Benchmark report: complexity, ECE, RAM, compression ratio
