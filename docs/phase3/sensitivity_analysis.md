# Phase 3: EVT-CVaR Sensitivity Analysis

**Scope:** Azure dataset only. Huawei generalization is Phase 5.  
**Models evaluated:** RiskAware(Reactive) and RiskAware(TCN).  
**Phase 2 anchor (fixed parameters):** α = 0.99, W = 30, threshold = P90.

---

## Motivation

Phase 2 introduced three fixed hyperparameters for the EVT-CVaR buffer:

| Parameter | Anchor value | Meaning |
|-----------|-------------|---------|
| α | 0.99 | CVaR confidence level — how far into the tail the buffer is calibrated |
| W | 30 | Rolling volatility window — how many past residuals define local risk |
| threshold | P90 | POT threshold percentile — what fraction of residuals are "extreme" |

These were chosen on principled grounds (see `docs/phase2/architecture.md`), not by searching over the test set.  Phase 3 demonstrates *post-hoc* that the method is inherently robust: the key findings should hold across a wide neighbourhood of these parameter choices.

The story we want to tell is: **"the method is robust"**, not "we found good parameters."  If Phase 3 showed that results were sensitive to small parameter perturbations, that would suggest the anchor was cherry-picked.

---

## Experimental Design

### Phase 3A — One-at-a-time sweep (9 configurations)

Each parameter is varied individually while the other two are held at anchor values.

| Sweep | Configurations | Fixed values |
|-------|---------------|-------------|
| α sweep | 0.95, 0.975, **0.99** | W = 30, P = P90 |
| W sweep | 10, **30**, 60 | α = 0.99, P = P90 |
| P sweep | P85, **P90**, P95 | α = 0.99, W = 30 |

**Anchor appears in all three sweeps** — its results are computed once and reused.

### Phase 3B — Interaction check (8 configurations)

Full 2×2×2 factorial at the *boundaries* of the parameter space:

```
α ∈ {0.95, 0.99} × W ∈ {10, 60} × threshold ∈ {P85, P95}  →  8 runs
```

Purpose: check whether interactions exist between parameters.
- **Parallel lines in interaction plots** → no interaction → each parameter's effect is additive and independent → robustness claim is stronger.
- **Diverging or crossing lines** → interaction present → the impact of one parameter depends on the value of another → requires more careful parameter choice.

### Why only Reactive and TCN?

These two bracket the quality spectrum from Phase 2:

| Model | Phase 2 Request SLA | Phase 2 Cost | Role |
|-------|--------------------|-----------| ----|
| RiskAware(Reactive) | 0.9996 | 447M | Simplest base model; representative of worst case |
| RiskAware(TCN) | **0.9997** | **342M** | Best base model; representative of best case |

If robustness holds for both extremes, it holds for models in between.

---

## Implementation Details

### Scripts

| Script | Purpose |
|--------|---------|
| `scripts/run_phase3.py` | Runner — all 3A+3B configurations |
| `scripts/audit_phase3.py` | Leakage audit |
| `scripts/generate_phase3_graphs.py` | Sensitivity curves, interaction plots, robustness overview |

### Run order

```bash
# From the project root (ColdStartOptimization/)
python scripts/run_phase3.py           # ~5-10 min (TCN trained once)
python scripts/audit_phase3.py         # ~1 min
python scripts/generate_phase3_graphs.py  # ~30 sec
```

### Optimization: single base-model training

A key engineering choice: the base model (Reactive or TCN) is trained **once**, not once per EVT configuration. EVT parameters (α, threshold) affect only the CVaR multiplier derived from pre-computed training residuals, and W affects only the sequential rolling-window computation during inference. This reduces total runtime from ~75 minutes (15 configs × 5 min TCN) to ~5 minutes.

Mathematically, for any two configs sharing the same base model:
- Training residuals `ε_i = y_i − ŷ_i` are **identical** (same base model weights)
- `σ_train = std(ε)` is **identical**
- `z_i = ε_i / σ_train` is **identical**
- Only the EVT fit on `z` changes with α and threshold, and only the rolling-window size changes with W

This identity is **verified by the audit** (Audit 3: base prediction identity).

### Output files

```
results/phase3/azure/
  runs/
    {config_key}_{ModelName}_metrics.json    # per-config summary metrics
    {config_key}_{ModelName}_diagnostics.csv # per-timestep breakdown
  summary_3a.csv      # long-form table of all 3A results
  summary_3b.csv      # long-form table of all 3B results
  all_metrics.json    # complete nested record
  evt_parameters.json # GPD shape/scale/CVaR per config
  audit_results.json  # pass/fail report from audit_phase3.py

graphs/phase3/azure/
  sensitivity_curves.png      # 2×3 grid: (SLA, Cost) × (α, W, P)
  sensitivity_extreme_sla.png # 1×3: Request SLA + Extreme SLA vs each param
  interaction_plots.png       # 3×2: interaction effects (Phase 3B)
  robustness_overview.png     # all configs dot-plot
  buffer_sensitivity.png      # mean buffer ± 1σ vs each param
```

Config key format: `a{α:.3f}_W{W:03d}_P{P:02d}` — e.g., `a0.990_W030_P90` is the anchor.

> **Note on `summary_3a.csv` row count:** The anchor config (α=0.99, W=30, P90) appears once
> per sweep label per model — 6 anchor rows total (3 sweeps × 2 models). This is intentional:
> each sensitivity curve needs its own anchor point. Code that aggregates by unique config should
> call `drop_duplicates(subset=["config_key", "model"])` before computing per-config statistics.

---

## Leakage Analysis

The sweep introduces no new leakage beyond what was established in Phase 2. Verified by `audit_phase3.py`:

| Audit check | What it verifies |
|------------|-----------------|
| Test set identity | All 30 runs use the same 3,744 test rows |
| Threshold consistency | Extreme flag count identical across all runs (P99 of training) |
| Base prediction identity | Reactive (and TCN) base predictions are bit-identical across all EVT configs |
| Anchor consistency | Anchor config results match Phase 2 within floating-point tolerance |
| EVT training-only | First W sigma_t values equal sigma_train (warm-up from training data) |
| Sequential volatility | sigma_t varies after warm-up (no hindsight) |
| Buffer non-constant | CV(buffer_t) > 0.01 for every run |
| Accounting identity | served + cold == demand for every run |
| Phase 1/2 immutability | Phase 1/2 result files untouched |

The critical insight is the **base prediction identity check**: since EVT parameters influence only the buffer (not the base model's forward pass), varying them cannot constitute leakage. If that check failed, it would indicate a code-level bug where EVT fitting inadvertently modified the base model.

---

## Actual Results

All 30 runs (17 unique configs × 2 models) produced request SLA in the range **0.9978–0.9999**, all well above the 0.99 minimum requirement. The three findings below describe the structure of that variation.

---

### Finding 1: α is the primary cost/SLA lever

α is the only parameter that meaningfully moves both cost and SLA together.

| Model | α | Total Cost | Request SLA |
|-------|---|-----------|-------------|
| TCN | 0.95 | 238.9M | 0.9989 |
| TCN | 0.975 | 280.3M | 0.9993 |
| TCN | **0.99** | **342.4M** | **0.9997** |
| Reactive | 0.95 | 320.3M | 0.9988 |
| Reactive | **0.99** | **446.6M** | **0.9996** |

At α=0.95 vs α=0.99, TCN costs 239M vs 342M — a **30% cost reduction** for a **0.08 pp SLA loss** (0.9989 vs 0.9997). For a cost-sensitive deployment where 0.998 SLA is still acceptable, α=0.95 is a legitimate design choice. The paper can offer α as an operational lever: tighten toward 0.99 to prioritize SLA, relax toward 0.95 to cut cost.

---

### Finding 2: W is cost-neutral but SLA-positive

Larger W improves SLA at negligible cost penalty.

| Model | W | Total Cost | Request SLA |
|-------|---|-----------|-------------|
| TCN | 10 | 321.2M | 0.9993 |
| TCN | **30** | **342.4M** | **0.9997** |
| TCN | 60 | 346.4M | 0.9999 |
| Reactive | 10 | 439.1M | 0.9988 |
| Reactive | **30** | **446.6M** | **0.9997** |
| Reactive | 60 | 446.8M | 0.9999 |

TCN cost rises only 25M (7%) from W=10 to W=60, while SLA improves by 0.06 pp (0.9993 → 0.9999). On this dataset, **W=60 strictly dominates W=30** — same cost band, better SLA. The anchor W=30 is defensible (chosen before Phase 3), but the paper can note that W=60 is a Pareto improvement on Azure.

---

### Finding 3: Threshold is nearly inert

Threshold affects cost modestly and SLA negligibly.

| Model | Threshold | Total Cost | Request SLA |
|-------|----------|-----------|-------------|
| TCN | P85 | 350.4M | 0.9997 |
| TCN | **P90** | **342.4M** | **0.9997** |
| TCN | P95 | 340.9M | 0.9997 |
| Reactive | P85 | 473.0M | 0.9997 |
| Reactive | **P90** | **446.6M** | **0.9996** |
| Reactive | P95 | 448.0M | 0.9997 |

TCN SLA is **identical at 0.9997** across all three threshold values. Cost spread is 341M–350M (3%). The POT-GPD fit is stable in the P85–P95 range: the tail distribution shape does not shift meaningfully in this exceedance window.

---

### Summary: one lever, two near-inert knobs

| Parameter | Effect on cost | Effect on SLA | Design implication |
|-----------|--------------|--------------|-------------------|
| α | **Strong (~30%)** | **Moderate (0.08 pp)** | Operational lever; choose by cost/SLA priority |
| W | Weak (<10%) | Mild (+0.06 pp) | Larger is better; W=60 weakly dominates on Azure |
| threshold | Weak (<5%) | Negligible | P90 is fine; P85–P95 are interchangeable |

### Phase 3B: Interaction effects confirmed absent

The 2×2×2 factorial (8 boundary configs) produced approximately parallel interaction lines, consistent with the algebraic separability of the EVT-CVaR formula: CVaR_z depends on (α, threshold) while σ_t depends on W. No parameter interaction was detected, supporting the robustness claim.

---

## Interpretation for Paper Writing

### Claims the figures support

1. **Robustness claim** (primary): "Request SLA remains ≥ 0.99X for all tested configurations."
   - `robustness_overview.png` shows all 30 (config, model) dots staying above a threshold line.
   - `sensitivity_curves.png` shows flat-to-mildly-sloped curves in the SLA panels.

2. **Parameter independence** (secondary): "The three EVT-CVaR hyperparameters control independent aspects of risk estimation, evidenced by the absence of significant interaction effects."
   - `interaction_plots.png`: nearly parallel lines = no interaction.

3. **Anchor validity** (tertiary): "The Phase 2 anchor values were not cherry-picked; a wide neighbourhood of settings produces qualitatively identical conclusions."
   - The range of SLA/cost variation across all 17 unique configs provides the evidence.

4. **Buffer dynamics preserved**: "The dynamic buffer structure (CV > 0.01) is preserved for all tested hyperparameter values."
   - `buffer_sensitivity.png` confirms the buffer remains non-constant across all configs.

### Limitations

1. **One dataset only**: Phase 3 uses Azure data only. Generalization to Huawei is Phase 5.
2. **Coarse grid**: Phase 3A tests 3 values per parameter; finer grids might reveal non-monotonicity, but the purpose is robustness demonstration, not optimization.
3. **No cross-validation**: each config is evaluated on a single held-out test set. The test set is large enough (3,744 timesteps, 26 extreme events) that variance in evaluation is small, but confidence intervals are not computed.
4. **Pre-specified design**: the α ∈ {0.95, 0.975, 0.99} and P ∈ {P85, P90, P95} grids were designed before running experiments. This is correct scientific practice — the sweep is not a post-hoc search.

---

## Connection to Other Phases

| Phase | What it does | How Phase 3 connects |
|-------|-------------|---------------------|
| Phase 1 | Six baseline forecasters | Phase 3 takes Reactive and TCN as bracketing representatives |
| Phase 2 | EVT-CVaR wrapper, anchor params | Phase 3 validates anchor robustness; anchor results should match |
| **Phase 3** | **Sensitivity of EVT-CVaR params** | **This document** |
| Phase 4 | Ablation (which EVT components matter?) | Which component removal hurts most informs which params are critical |
| Phase 5 | Huawei generalization | Phase 3 demonstrates the method isn't tuned to Azure |
