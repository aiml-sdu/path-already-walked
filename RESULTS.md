# Phase 2 results: GPT-2 small, IOI, pre-registered (2026-11-16)

- **Run:** `poc.py`, 545 s on an M3 Pro (MPS).
- **PREREG.md:** SHA-256 `5c14bd18…` was recorded before the run, verified at launch and stored in `results/results.json`.
- **Sanity checks:** the manual z equals the model's hook_z (difference 0.0). The mean held-out logit difference is 3.51, which is in line with GPT-2 small on IOI.

## Pre-registered verdicts (10% density)

| | Statistic | Null | Verdict |
|---|---|---|---|
| **H1 rich club (a)** | Φ_norm(k) = 1.00 (bootstrap 95% CI [0.97, 1.00]), Φ = 0.21 | null mean 0.21, p = 1.0 | **fail** (Φ_norm ≤ 1) |
| **H1 rich club (b)** | overlap \|R ∩ I\| = 1 of 25 | hypergeometric p = 0.99 | **fail** |
| **H2 modularity** | AMI = −0.016 (CI [−0.11, −0.02]), 31 communities, Q = 0.19 | permutation 97.5th percentile 0.172, p = 0.57 | **fail** |

**Both pre-registered predictions failed.** They are reported as failures.

## Sensitivity (pre-registered as report-only, not used for the verdict)

| Density | Φ_norm, p | Overlap, p | AMI, p |
|---|---|---|---|
| 5% | 1.00, 1.0 | 1, 1.0 | −0.016, 0.67 |
| 20% | 0.98, 1.0 | 4, 0.85 | 0.183, **0.022** |

H2 would pass at 20% density. That is exactly the threshold dependence the paper warns about, and it must **not** be reported as support.

## Edge validation (descriptive)

Spearman between EC and finite path patching, over 20 edges: **ρ = 0.38**, bootstrap 95% CI over prompts [0.36, 0.45]. The 10 strongest EC edges all run from layer 0 or 1 into layers 2 and 3, with EC around 4 to 7 × 10³. The 10 random retained edges have EC around 50 to 230. Their finite effects are similar in size (0.1 to 0.5, with one at 2.0). The proxy ranks edges only weakly in the way path patching does.

## Post-hoc diagnosis (not pre-registered; labelled as such wherever it is reported)

1. **The graph is a scale artefact of early layers.** Mean EC by source layer is 161 (layer 0), 102 (layer 1), then 5 to 8 for all later layers. With the top-10% threshold, 927 of the 950 edges leave layers 0 and 1. The "rich club" is therefore simply layer 0 plus part of layers 1 and 3. In raw weight coordinates, the unnormalised Jacobian norm is dominated by the output scale of early heads, not by routing.
2. **The null cannot move Φ for this club.** Swaps confined to a layer pair keep every node's in- and out-degree **within that block**. When the club consists of whole layers, as here, the number of edges inside the club is then fixed by construction: every null gives Φ = 0.21 exactly, and the test has no power. The null in PREREG.md row 1 (Table 3, row 1 of the paper) is the paper's own recommendation, so this is a finding about the contract and not only about GPT-2.
3. **Consequence for the paper.** The worked example shows a concrete case of the paper's central warning. The same model and task give a degenerate rich club under one edge normalisation and null, and a threshold-dependent modularity result. A follow-up would pre-register a scale-normalised edge (for example EC divided by the sender's output norm, or a correlation-style normalisation) and a null that preserves only global degree and layer order. **That would be a new pre-registration, not a re-analysis of this one.**

## Files

`PREREG.md`, `PREREG.sha256`, `poc.py`, `requirements.txt`, `results/results.json`, `results/ec_mean.npy`, `results/ec_per_prompt.npy`, `run.log`.
