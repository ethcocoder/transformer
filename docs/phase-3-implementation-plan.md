# Phase III — ARC — Implementation Plan

**Depends on:** Phase II checkpoint  
**Blocks:** Phase IV (TESSERAE)  
**Spec reference:** project.md § Phase III — 7 components

---

## 1. Objective

Activate **reasoning channel r** with a universal attractor-based core:

- One recurrence, multiple modes via control vector α_t (k=4)
- Attractor bank Ξ, selection π_t, basin geometry Λ_m
- **4-pass scan:** c → e → r → m (reordered per spec FPSS)
- Cross-channel: RCM, ERUG, MRDI
- Reasoning-conditioned decoding temperature τ(r_t)

---

## 2. Component → Code Mapping

| Component | Module |
|-----------|--------|
| **1 RSE** — CRR, ATS, BCC | `src/arc/rse.cpp` |
| **2 ALE** — LAB, ASA, GBS | `src/arc/ale.cpp` |
| **3 RCVG** — MMD, MBI, dissipation | `src/arc/rcvg.cpp` |
| **4 MC-FWS** — MSPG, INS, CSRG | `src/arc/mc_fws.cpp` |
| **5 CCRC** — CRI, RCM, ERUG, MRDI | `src/arc/ccrc.cpp` |
| **6 ASK-R** — 4-pass scheduler, BAR | `src/arc/ask_r.cpp` |
| **7 ROI** — RCLR, TEM | `src/arc/roi.cpp` |

---

## 3. Four-Pass Scan Schedule

```
Pass 1: LENS → c_{1:n}
Pass 2: AUREUM epistemic → e_{1:n}  (uses m_{t-1})
Pass 3: compute (a^r, b^r) from c,e,m,α; scan → r_{1:n}
Pass 4: meta → m_{1:n}  (uses c,e)
```

**Bias decomposition (ATS):**
`b^r = (1 - a^r) ⊙ ξ^eff + γ_t`  
where `ξ^eff = π^T Ξ` from attractor bank.

---

## 4. Implementation Steps

### Step 1 — ALE: Attractor Bank (Week 1)
- `Ξ ∈ R^{M×D_r}`, M=64 default
- ASA: `π = softmax(W_π [c;e;α] + b)`
- GBS: per-attractor Cholesky `Λ_m` (precision)
- Init: K-means++ on random forward r activations

### Step 2 — RCVG (Week 1–2)
- `α^meta = W_α @ m_{t-1} + b_α`
- `α = softmax(α^meta + α^prompt)` — prompt from instruction embed or zeros
- Mode dissipation: `ε^eff = Σ_j α_j ε̄_j` with fixed table:
  - Deduce 0.50, Plan 0.15, Create 0.02, Analogize 0.10

### Step 3 — MC-FWS + BCC (Week 2)
- `a^r = 1 - ε^eff(α) ⊙ β_t`
- INS: optional Gaussian noise for create mode (`α_3 * ν`)
- CSRG: partition D_r into K_r scales like MSSP

### Step 4 — RSE scan (Week 2–3)
- Reuse Phase 0 scan with r-specific (a,b)
- MGP on r^raw before CCRC

### Step 5 — CCRC (Week 3)
- RCM: `c_out = c + W_cr @ r + b_cr`
- ERUG: scale ε^eff by `κ + (1-κ)*η_explore`
- MRDI: `π ← softmax(log π + W_mr @ d)`

### Step 6 — ROI (Week 3–4)
- Logits with `c_out`
- Temperature: `τ = τ_0 * σ(w_τ^T r + b_τ)`
- TEM: `H^reason = -Σ π_m log π_m`

### Step 7 — ASK-R integration (Week 4)
- `ArcPassScheduler` replaces Aureum 3-pass with 4-pass
- BAR: reverse adjoint scan for r (C kernel)

### Step 8 — Training (Week 5–7)
- Loss: `L_pred + λ_c L_CSL + λ_d L_MCL + λ_r L_reason + λ_s L_sparse`
- `L_reason`: MSE(r_t, r_target) if CoT traces available; else skip
- Curriculum per spec: freeze c,e,m → unfreeze → mode supervision → full

---

## 5. Mode Supervision (optional)

If reasoning-type labels exist:
- Cross-entropy on α vs mode class
- **Python:** map task types to mode indices

---

## 6. Tests

| Test | Assert |
|------|--------|
| Attractor convergence | ||r_t|| bounded (Theorem 2) |
| Mode switch continuity | no jump in r at α change |
| π sparse | entropy decreases with L_sparse |
| 4-pass linear | time ∝ n |
| Deduce vs Create | τ lower when ε high (sharper vs flatter logits) |

---

## 7. Exit Criteria

- [ ] r channel fully recurrent (not placeholder)
- [ ] α switches modes; measurable τ/logit shape change
- [ ] Attractor selection π peaks on coherent prompts
- [ ] Full 4-channel forward/backward stable for 8k tokens
- [ ] Phase II metrics (ECE, MCL) maintained

**Estimated duration:** 5–6 weeks after Phase II
