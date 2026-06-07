# Phase IV — TESSERAE — Implementation Plan

**Depends on:** Phase III dense checkpoint  
**Blocks:** Phase V (CONVERGENCE)  
**Spec reference:** project.md § Phase IV — 7 components

---

## 1. Objective

Compress trained Aurelis for **edge deployment**:

- Store ψ (hypernetwork) + `{z^(l)}` + TR cores — not full weights
- Dynamic layer synthesis (D3L): peak RAM < 50MB (target config)
- Epistemic-guided INT4/INT8 quantization (EPQE)
- Distillation preserves epistemic frames (EFD)
- Flat binary format ODFFB

**Inference path:** C++ only (no Python on device).

---

## 2. Component → Code Mapping

| Component | Module | Language |
|-----------|--------|----------|
| **1 INWS** — LCR, CBHC, LTMH | `src/tesserae/inws.cpp` | C++ |
| **2 TRFC** — RATD, CTR, STS | `src/tesserae/trfc.cpp`, `src/c/tensor_ring.c` | C |
| **3 EPQE** — DMAS, CGFC, MPQ | `src/tesserae/epqe.cpp`, `src/c/quant_int4_int8.c` | C |
| **4 EDS** — ODFFB, D3L, ACRP | `src/tesserae/eds.cpp` | C++ |
| **5 EFD** — FAKT, MMPL, ABDP | `src/tesserae/efd.cpp` | C++ |
| **6 CVK** — FBC, SBA, ICB | `src/tesserae/cvk.cpp` | C++ |
| **7 P4IL** — SER, QGI, CPCC | `src/tesserae/p4il.cpp` | C++ + Python CLI |

---

## 3. Storage Model

```
Stored artifact B =
  [ ψ_q4                           ]  shared hypernetwork ~1M params
  [ z^(1)_q8 ... z^(L)_q8          ]  L × k scalars (k=256..1024)
  [ TR cores for large M matrices  ]  per-layer structured
  [ L_m, Ξ, R globals FP16         ]  geometry + attractors
```

**Synthesis (per layer, per forward):**
```
θ^(l) = INR_ψ(z^(l), γ(i), t_τ)   OR   matmul via TR cores
→ run LENS/AUREUM/ARC with materialized θ
→ evict θ from RAM
```

---

## 4. Implementation Steps

### Step 1 — INWS Hypernetwork (Week 1–2)
- 3-layer MLP: `[z; PE(i); t_τ] → θ_i`
- Sinusoidal PE over flattened param index
- Layer-type embedding `t_τ` (LENS, AUREUM, ARC)
- **Training stage 1:** MSE reconstruct dense teacher weights

### Step 2 — CPCC Converter (Week 2)
- **Python:** `python/tools/cpcc_convert.py phase3.ckpt → tesserae_init/`
- Meta-learn ψ, optimize z^(l) per layer
- Export sidecar manifest (shapes, types)

### Step 3 — TRFC (Week 3)
- Reshape W ∈ R^{m×n} to order-d tensor
- TR cores G_p ∈ R^{r×N_p×r}
- CTR: matvec without full materialize — **C:** `tensor_ring_matvec.c`
- STS: SVD per unfolding, rank for 99% energy

### Step 4 — EPQE Quantization (Week 3–4)
- DMAS: affinity ω_k^(l) from z^(l) and domain centroid
- CGFC: freeze z gradient for mature (layer, domain) pairs
- MPQ: INT8/INT4 with STE — **C:** `quant_dequant_ste.c`
- Bit policy per spec table (mature→INT4, metric L_m→FP16)

### Step 5 — EFD Distillation (Week 4–5)
- Run teacher (dense) and student (compressed) on same batch
- `L_FAKT`: MSE on c,e,r,m with metric-weighted norms
- `L_MMPL`, `L_ABDP` on G_m and Ξ

### Step 6 — D3L Runtime (Week 5–6)
```cpp
class D3LLoader {
  void load_layer(int l);
  void synthesize_weights(int l);
  void evict_layer(int l);
  float peak_ram_mb() const;
};
```
- Single-layer weight buffer reused
- ACRP: gradient checkpoint every √L layers (training only)

### Step 7 — ODFFB Baker (Week 6)
- Binary layout: header magic `AURL`, version, offsets
- **Python:** `bake_odffb.py` → `model.aurl`
- **C++:** `mmap` read for inference

### Step 8 — CVK Verification (Week 7)
- FBC: ΔPPL < ε, ΔECE < ε vs teacher
- SBA: total size ≤ 500MB
- ICB: variational MI lower bound on z

---

## 5. Training Stages (Python)

| Stage | Steps | Loss |
|-------|-------|------|
| 1 Warmup | 0–20% | L_recon |
| 2 QAT | 20–70% | L_pred + L_FAKT + STE |
| 3 TR prune | 70–90% | fine-tune TR ranks |
| 4 Bake | 90–100% | export + CVK |

---

## 6. Edge Inference Binary

```
aurelis_infer.exe  --model model.aurl  --prompt "..." 
```

Pure C++: D3L + scan kernels + no autodiff.

---

## 7. Tests

| Test | Assert |
|------|--------|
| INR recon | per-layer MSE < 1% of weight variance |
| TR matvec | matches dense matvec error < 1e-2 |
| INT4 STE | backward reaches z and ψ |
| Peak RAM | < 50MB on reference D,L config |
| FBC | ΔPPL < 5% rel, ΔECE < 0.02 |

---

## 8. Exit Criteria

- [ ] CPCC converts Phase III ckpt to TESSERAE format
- [ ] D3L inference runs without dense weights in memory
- [ ] ODFFB loads on clean C++ runtime
- [ ] Epistemic κ calibration within ε of teacher
- [ ] Compression ratio ≥ 100× on dev model

**Estimated duration:** 7–8 weeks after Phase III
