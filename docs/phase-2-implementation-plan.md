# Phase II — AUREUM — Implementation Plan

**Depends on:** Phase I checkpoint  
**Blocks:** Phase III (ARC)  
**Spec reference:** project.md § Phase II — 7 components

---

## 1. Objective

Activate **explicit self-knowledge**:

- Epistemic recurrence on **e** channel (ESE)
- Competence head κ_t calibrated to accuracy
- Meta channel **m** as domain coordinate on Riemannian manifold
- Domain transfer via geodesic parallel transport (DTC)
- **3-pass scan:** c → e → m

---

## 2. Component → Code Mapping

| Component | Module |
|-----------|--------|
| **1 ESE** — KA, UVH, EFWS | `src/aureum/ese.cpp` |
| **2 CQ** — κ, calibration, OOD | `src/aureum/cq.cpp` |
| **3 KMG** — metric G_m, DCI, MGP-M | `src/aureum/kmg.cpp` |
| **4 DTC** — atlas, geodesic, PTG | `src/aureum/dtc.cpp` |
| **5 CCC** — CEI, ECM, MEG | `src/aureum/ccc.cpp` |
| **6 SRLS** — losses | `src/aureum/srls.cpp` + `python/train/losses_phase2.py` |
| **7 PII** — router, MOGB, scan schedule | `src/aureum/pii.cpp` |

---

## 3. Scan Scheduler (PII)

```
Pass 1: LENS scan → c_{1:n}
Pass 2: parallel compute (a^e, b^e) from c_t, m_{t-1}; scan → e_{1:n}
Pass 3: parallel compute (a^m, b^m) from c_t, e_t; scan → m_{1:n}
```

**C++ class:** `AureumPassScheduler` orchestrates three `scan_blelloch` calls per layer.

**Dependency note:** e pass uses `m_{t-1}` — shift m by one or compute m scan with delayed injection (spec: external control signal).

---

## 4. Implementation Steps

### Step 1 — Load Phase I checkpoint (Week 1)
- `CheckpointLoader` restores LENS weights
- Freeze LENS for first 10% steps (Python trainer flag)

### Step 2 — ESE + EFWS (Week 1–2)
- Same diagonal recurrence as LENS on D_e dims
- Injection: `silu(W_ec @ c_t + b_ec)`
- UVH: `σ = softplus(W_σ @ e_t + b_σ)`

### Step 3 — CQ (Week 2)
- `κ = σ(w_κ^T e_t + b_κ)`
- CTG: binary `y_acc` from argmax logits vs target
- `L_CSL = BCE(κ, y_acc)`
- EBD: `ω = σ(λ(σ̄ - τ_ood))`

### Step 4 — KMG (Week 2–3)
- `G_m = L_m L_m^T + εI` — Cholesky param (reuse `linalg_small.c`)
- DCI recurrence on m channel
- MGP-M: project m onto G_m-sphere
- Geodesic distance: `sqrt((d_i-d_j)^T G_m (d_i-d_j))`

### Step 5 — DTC (Week 3)
- DAC: online EMA centroids per domain k (Python provides labels or pseudo-labels)
- GFA: geodesic direction `v_{s→t}`
- PTG: gated content perturbation for transfer inference

### Step 6 — CCC (Week 3–4)
- ECM: `c_mod = κ*c + (1-κ)*W_fallback @ e`
- MEG: modulate `a^e` by `σ(W_me' @ m_{t-1})`

### Step 7 — SRLS + MOGB (Week 4)
- MCL: batch contrastive on mean d̄ per sequence — **Python** loop over batch
- MRL: `||L_m L_m^T - I||_F² + λ_det (log det G_m)²`
- MCER: entropy of `softmax(W_dom @ d_t)`
- MOGB: PCGrad — start with fixed loss weights; add PCGrad if conflicts

### Step 8 — Training curriculum (Week 5–6)
- 0–20%: `L_pred + λ_c L_CSL`
- 20–60%: + `L_MCL`
- 60–100%: + `L_MRL + L_MCER`
- LR = 0.1 × Phase I peak

---

## 5. Python Trainer Additions

```python
# python/train/train_phase2.py
trainer.load_checkpoint("phase1.ckpt")
trainer.freeze_lens(steps=int(0.1 * total_steps))
trainer.set_loss_weights({"csl": 0.1, "mcl": 0.05, "mrl": 0.01, "mcer": 0.001})
```

Domain labels: JSON sidecar or folder-per-domain dataset.

---

## 6. Tests

| Test | Assert |
|------|--------|
| 3-scan vs sequential | e,m match sequential recurrence |
| Calibration | ECE(κ) decreases over training |
| MCL pull | same-domain d closer than diff-domain |
| ECM gating | low κ increases fallback contribution |
| Complexity | 3× Phase I time, still O(n) vs sequence |

---

## 7. Exit Criteria

- [ ] κ correlates with accuracy (ECE < 0.08 dev)
- [ ] Domain clusters visible in 2D PCA of d_t
- [ ] PTG improves cross-domain transfer on held-out eval
- [ ] r channel still L2-regularized only
- [ ] Phase I perplexity not regressed > 5%

**Estimated duration:** 5–6 weeks after Phase I
