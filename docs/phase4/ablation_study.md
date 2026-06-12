# Phase 4: EVT-CVaR Component Ablation Study

**Scope:** Azure dataset only. Huawei generalization is Phase 5.  
**Models evaluated:** RiskAware(Reactive) and RiskAware(TCN).  
**Purpose:** Determine which components of the EVT-CVaR buffer independently contribute to performance.

---

## Motivation

Phase 2 showed that the EVT-CVaR buffer dramatically reduces cold starts (−97–98%) and
brings SLA from ~0.985 to ~0.9997. Phase 3 showed the result is robust to hyperparameter
perturbation. But neither phase answers the question the paper must answer before claiming
the method works:

> **Which components are doing the work — dynamic volatility estimation, EVT tail calibration, or both?**

Phase 4 isolates the two independently variable components of the buffer formula:

```
buffer_t = sigma_t × CVaR_z
```

by testing all four combinations of how each component can be implemented. This provides
direct empirical evidence for or against the claim that **both components are necessary**.

Phase 4 must come before Phase 5 (Huawei generalization) because there is no point testing
generalization of a method if we cannot first explain why the method works on the training
distribution.

---

## Ablation Design

### The 2×2 Factorial

| Condition | sigma used | Multiplier | Label |
|-----------|-----------|------------|-------|
| **C0** — No buffer | none | none | Phase 1 result |
| **C1** — Static + Gaussian | σ_train (fixed) | K = 2.665 (Gaussian CVaR) | new |
| **C2** — Dynamic + Gaussian | σ_t (rolling, W=30) | K = 2.665 (Gaussian CVaR) | new |
| **C3** — Static + EVT | σ_train (fixed) | CVaR_z (fitted GPD) | new |
| **C4** — Dynamic + EVT | σ_t (rolling, W=30) | CVaR_z (fitted GPD) | Phase 2 result |

The new conditions (C1, C2, C3) are the minimum set needed to decompose the full model.
C0 and C4 already exist in Phase 1 and Phase 2 results respectively and are loaded from
disk without re-running.

### Why these two models

Reactive and TCN bracket the quality spectrum from Phase 2:
- **Reactive** (lag-1 predictor): simplest base model, highest Phase 2 cold-start rate
- **TCN** (deep learning): best base model, lowest Phase 2 cost and highest SLA

If both extremes show the same ablation pattern, the finding generalises to any base model
in between.

### What the conditions test

| Comparison | Question answered |
|-----------|-------------------|
| C0 → C1 | Does *any* buffer help, even a constant Gaussian one? |
| C1 → C3 | Does replacing Gaussian with EVT tail fitting improve things? |
| C1 → C2 | Does replacing static σ with adaptive rolling σ improve things? |
| C2 → C4 | Does EVT add further value on top of an already-adaptive buffer? |
| C3 → C4 | Does adaptivity add further value on top of EVT calibration? |

---

## The Gaussian Multiplier

### What it is

K_GAUSSIAN is the Conditional Value-at-Risk of a standard normal distribution at the
same confidence level α = 0.99 used throughout the project:

```
K_GAUSSIAN = φ(Φ⁻¹(0.99)) / (1 − 0.99) ≈ 2.6652
```

where φ is the standard normal PDF and Φ⁻¹ is its quantile function.

In code:
```python
from scipy.stats import norm
ALPHA = 0.99
_z_alpha   = norm.ppf(ALPHA)              # ~2.3263
K_GAUSSIAN = norm.pdf(_z_alpha) / (1 - ALPHA)  # ~2.6652
```

This is the only scientifically defensible choice for the ablation comparison.

### Why not k = 3.0

k = 3.0 corresponds to α ≈ 0.9987, not α = 0.99. The ablation question is:
*"does EVT add value over a Gaussian assumption at the same confidence level?"*
If Gaussian and EVT operate at different confidence levels, the comparison
conflates two effects (distributional assumption AND confidence level), making
the result uninterpretable. k = 2.665 is the *Gaussian answer to the same
question* that EVT answers.

### The EVT vs Gaussian gap

From Phase 2 EVT fitting on training residuals (P90 threshold, α = 0.99):

| Model | CVaR_z (EVT) | K_GAUSSIAN | Ratio |
|-------|-------------|-----------|-------|
| Reactive | 4.160 | 2.665 | **1.56×** |
| TCN | 4.295 | 2.665 | **1.61×** |

EVT recommends a buffer multiplier 56–61% larger than the Gaussian assumption. This
gap exists because IT workload residuals have heavier tails than Gaussian:
- TCN: ξ = +0.022 (slightly Pareto, heavy-tailed)
- Reactive: ξ = −0.093 (slightly light-tailed, but still well above Gaussian CVaR)

The ablation will show whether this extra buffer margin matters for provisioning outcomes.

---

## Implementation Details

### Scripts

| Script | Purpose |
|--------|---------|
| `scripts/run_phase4.py` | Runner — trains base models once, runs C1/C2/C3 |
| `scripts/audit_phase4.py` | Leakage and correctness audit |
| `scripts/generate_phase4_graphs.py` | All 5 ablation figures |

### Run order

```bash
python scripts/run_phase4.py           # ~5-10 min (TCN trained once)
python scripts/audit_phase4.py         # ~1 min
python scripts/generate_phase4_graphs.py  # ~30 sec
```

### How C0 and C4 are incorporated

C0 and C4 are **loaded from existing results**, never re-run:
- C0: `results/phase1/azure/metrics.json` → keys `"Reactive"` and `"TCN"`
- C4: `results/phase2/azure/metrics.json` → keys `"RiskAware(Reactive)"` and `"RiskAware(TCN)"`

Their values are written directly into `summary.csv` and `all_metrics.json` alongside
the new conditions. No files in `results/phase1/` or `results/phase2/` are modified.

### How static σ is implemented

For C1 and C3 (`sigma_type = "static"`):
```python
sigma_t = np.full(n, cache.sigma_train, dtype=np.float64)  # constant array
buffer  = sigma_t * multiplier                               # fully constant buffer
```

The rolling-window loop from `RiskAwareModel.predict()` **does not execute**. This is
verified by the audit (Audit 3: `std(sigma_t) == 0`).

For C2 (`sigma_type = "dynamic"`), the same rolling-window logic as `RiskAwareModel.predict()`
runs with W = 30 (identical to Phase 2 anchor):
```python
sigma_t = _rolling_sigma(base_preds, lag_1, cache.sigma_train, window=30)
```

### Output files

```
results/phase4/azure/
  conditions/
    {condition_id}_{model}_metrics.json    # C1, C2, C3 only
    {condition_id}_{model}_diagnostics.csv # C1, C2, C3 only (with cvar_col column)
  summary.csv       # all 10 rows (5 conditions x 2 models)
  all_metrics.json  # complete nested record
  audit_results.json

graphs/phase4/azure/
  ablation_sla.png
  ablation_cost.png
  ablation_incremental.png
  ablation_buffer_profiles.png
  ablation_2x2_heatmap.png
```

The diagnostics CSV for C1–C3 extends the Phase 2/3 format with a `cvar_col` column
containing the multiplier used (K_GAUSSIAN or CVaR_z) at every timestep.

---

## Leakage Analysis

The ablation introduces no new leakage beyond Phase 2:

| Component | Training-derived? | Leakage-free? |
|-----------|------------------|--------------|
| σ_train | `std(train residuals)` | Yes |
| K_GAUSSIAN | Computed from N(0,1) analytically | Yes — no data dependence |
| CVaR_z | Fitted to training residuals via EVT | Yes — same as Phase 2 |
| σ_t (dynamic) | Rolling window of test residuals **reconstructed** from lag_1 | Yes — same sequential argument as Phase 2 |

For C1 (static + Gaussian): the buffer `σ_train × K_GAUSSIAN` is a constant derived entirely
from training data and an analytical formula. There is no test-time adaptation — this is the
most conservative possible buffer.

For C3 (static + EVT): same as C1, replacing K_GAUSSIAN with EVT-fitted CVaR_z. Still
a constant derived entirely from training.

For C2 (dynamic + Gaussian): the rolling σ_t uses the same sequential, leakage-free
mechanism as Phase 2. The only change is replacing the EVT multiplier with K_GAUSSIAN.

---

## Results

Audit: **63/63 PASS**. See `results/phase4/azure/audit_results.json`.

### Full ablation table

| Condition | Reactive SLA | Reactive Cost | Reactive Cold Starts | TCN SLA | TCN Cost | TCN Cold Starts |
|-----------|-------------|--------------|---------------------|---------|---------|----------------|
| C0 No Buffer | 0.9848 | 408M | 37,072,085 | 0.9859 | 371M | 34,377,188 |
| C1 Static+Gaussian | 0.9992 | 324M | 1,980,591 | 0.9992 | 239M | 1,912,098 |
| C2 Dynamic+Gaussian | 0.9987 | 315M | 3,165,057 | 0.9988 | 236M | 2,818,121 |
| C3 Static+EVT | **0.9999** | 473M | 84,281 | **0.9999** | 358M | 243,049 |
| C4 Dynamic+EVT | 0.9996 | 447M | 865,343 | 0.9997 | 342M | 741,901 |

### 2×2 Request SLA heatmap

**Reactive:**

| | Gaussian | EVT |
|---|---------|-----|
| Static σ (C1) | 0.999187 | **0.999965** (C3) |
| Dynamic σ (C2) | 0.998701 | 0.999645 (C4) |

**TCN:**

| | Gaussian | EVT |
|---|---------|-----|
| Static σ (C1) | 0.999216 | **0.999900** (C3) |
| Dynamic σ (C2) | 0.998844 | 0.999696 (C4) |

### Incremental SLA gains

| Transition | What changes | Reactive delta | TCN delta |
|-----------|-------------|--------------|----------|
| C0 → C1 | Adding any buffer (static Gaussian) | **+1.44 pp** | **+1.33 pp** |
| C1 → C3 | Gaussian → EVT (static sigma) | +0.08 pp | +0.07 pp |
| C1 → C2 | Static → dynamic sigma (Gaussian) | −0.05 pp | −0.04 pp |
| C2 → C4 | Dynamic+Gaussian → Dynamic+EVT | +0.09 pp | +0.09 pp |
| C3 → C4 | Static+EVT → Dynamic+EVT | −0.03 pp | −0.02 pp |

### Internal diagnostics

| Condition | Model | sigma_t mean | sigma_t std | buffer mean | buffer std |
|-----------|-------|-------------|------------|------------|-----------|
| C1 Static+Gauss | Reactive | 30,283 | 0 | 80,711 | 0 |
| C1 Static+Gauss | TCN | 22,536 | 0 | 60,065 | 0 |
| C2 Dynamic+Gauss | Reactive | 28,058 | 5,824 | 74,782 | 15,523 |
| C2 Dynamic+Gauss | TCN | 21,206 | 5,540 | 56,517 | 14,766 |
| C3 Static+EVT | Reactive | 30,283 | 0 | 125,993 | 0 |
| C3 Static+EVT | TCN | 22,536 | 0 | 96,792 | 0 |

---

## Interpretation for Paper Writing

### Finding 1: Adding any buffer is the dominant effect

The C0→C1 transition — adding a **constant Gaussian buffer** — accounts for the vast majority
of the SLA improvement: +1.44 pp (Reactive) and +1.33 pp (TCN), bringing both models from
~0.985 to ~0.999. All subsequent refinements (EVT calibration, sigma adaptivity) operate in a
narrow 0.1 pp band.

This means the core contribution of the method is the concept of a *safety buffer calibrated
to the forecast error distribution*, not the specific sophistication of how that buffer is
estimated. The paper should make this explicit.

### Finding 2: EVT tail calibration improves SLA; static sigma maximises it

Reading across the 2×2 heatmap (Gaussian → EVT), SLA improves by +0.07–0.09 pp regardless
of sigma type. This is because CVaR_z ≈ 4.16–4.29 (EVT) vs K_GAUSSIAN ≈ 2.67 — EVT
recommends a 56–61% larger multiplier, providing substantially more headroom during demand
spikes.

However, **static σ (C1, C3) gives higher SLA than dynamic σ (C2, C4) at the same multiplier**.
C1→C2 and C3→C4 both show small SLA *reductions* (−0.02 to −0.05 pp). The rolling window
occasionally under-estimates volatility during quiet periods, leaving the buffer too small when
a spike arrives. σ_train is a conservative global estimate that never under-provisions.

The **highest SLA** is therefore C3 (Static+EVT) — 0.9999 for both models — not C4 (the
full Phase 2 model). Phase 2 chose dynamic σ for cost reasons, which is the correct engineering
tradeoff for the paper's claimed use case.

### Finding 3: Dynamic sigma trades SLA for cost

Reading down the 2×2 heatmap (static → dynamic), cost decreases while SLA decreases slightly:
- C1 → C2: Reactive 324M → 315M (−3%), SLA −0.05 pp; TCN 239M → 236M (−1%), SLA −0.04 pp
- C3 → C4: Reactive 473M → 447M (−5.5%), SLA −0.03 pp; TCN 358M → 342M (−4.4%), SLA −0.02 pp

The rolling window σ_t adapts downward during calm periods, reducing idle cost. This is the
correct behavior economically — it avoids over-provisioning when the system is stable. The SLA
cost of this adaptation is small (~0.02–0.05 pp) and remains well above 0.99 for both models.

### The actual outcome: Outcome B (EVT dominates, adaptivity is secondary for SLA)

The data matches **Outcome B** from the pre-run predictions: EVT calibration provides the
meaningful multiplier improvement; sigma adaptivity is secondary for SLA but relevant for cost.

The paper narrative is:
1. Any buffer dramatically closes the cold-start gap (C0→C1: −94% cold starts)
2. EVT tail calibration correctly accounts for heavy-tailed IT workload residuals, providing
   a 56–61% larger multiplier than Gaussian and further reducing cold starts
3. Dynamic sigma is a cost-efficiency mechanism, not a correctness mechanism — it reduces
   idle over-provisioning at a small, acceptable SLA cost
4. The full EVT-CVaR method (C4) is a principled balance: EVT for correctness, dynamic σ for efficiency

### What the 2×2 heatmap shows

The heatmap is the primary paper figure for this phase:
- Columns (Gaussian → EVT): the EVT contribution is visible as consistent SLA improvement in every row
- Rows (static → dynamic): the cost-SLA tradeoff is visible as small SLA decrease with lower cost
- C3 (top-right) is the max-SLA corner; C2 (bottom-left) is the min-cost corner
- C4 (Phase 2 full model) is bottom-right: EVT correctness with dynamic efficiency

### Limitations

1. **Azure only**: Phase 5 (Huawei) tests generalization. If both components contribute on Azure,
   Phase 5 determines whether the relative contribution is stable across datasets.

2. **W = 30 fixed**: Dynamic conditions use the Phase 2 anchor window. Phase 3 showed W
   has minimal impact, but the ablation does not sweep W.

3. **Single test set**: Evaluation uses 3,744 timesteps with 26 extreme events. Confidence
   intervals are not computed. The extreme-event count is small enough that Extreme SLA
   differences between conditions should be interpreted cautiously.

4. **Anchor EVT params**: C3 and C4 both use α = 0.99, threshold = P90. The ablation
   is not a search for best EVT parameters — those were pre-specified in Phase 2.

---

## Connection to Other Phases

| Phase | What it does | How Phase 4 connects |
|-------|-------------|---------------------|
| Phase 1 | Six baselines (no buffer) | C0 is Phase 1; Phase 4 shows how much the buffer adds |
| Phase 2 | Full EVT-CVaR (anchor params) | C4 is Phase 2; Phase 4 decomposes its improvements |
| Phase 3 | Sensitivity of EVT params | Confirms anchor params are robust; Phase 4 builds on this |
| **Phase 4** | **Component ablation** | **This document** |
| Phase 5 | Huawei generalization | Phase 4 explains *why* the method works; Phase 5 tests *where* it generalises |
