# Pre-registration: graph-contract proof of concept on GPT-2 small

Written 2026-11-16, before any model output was computed on the task prompts. No statistic below may be changed after the run. Any deviation must be reported as a deviation, with the reason.

Two translations from Table 2 of the paper are tested on one shared graph, each at Bonferroni-corrected α = 0.025 (one-sided).

## Hypotheses and failure criteria (verbatim from Table 2)

**H1. Rich-club ordering (priority Direct)**
- Claim to test: "High-degree heads, neurons, or features interconnect above degree-preserving nulls and overlap with importance sets above chance."
- Would fail if: "$\Phi_\mathrm{norm}(k)\le1$ or overlap is random-level under a held-out importance test."

**H2. Modularity (priority Moderate)**
- Claim to test: "Communities align with independently labelled functions such as induction, copying, or refusal routing."
- Would fail if: "Alignment is random-level under labelled probes."

## Operationalisation of the failure criteria (fixed now)

- **H1 fails** if either of the following holds:
  - (a) Φ_norm(k) ≤ 1, or Φ(k) of the observed graph is not above the 97.5th percentile of Φ(k) over the null ensemble (one-sided p ≥ 0.025);
  - (b) the overlap between the rich club and the held-out importance set is not significant under a one-sided hypergeometric test at p < 0.025.
  
  H1 passes only if both (a) and (b) pass.
- **H2 fails** if the adjusted mutual information (AMI) between the detected communities and the head labels, computed on the labelled heads, is not above the 97.5th percentile of AMI under 1000 random permutations of the labels over those heads (one-sided p ≥ 0.025).

## Graph contract

| Item | Declaration |
|---|---|
| Model | GPT-2 small (`gpt2`), TransformerLens 3.9.0 `HookedTransformer.from_pretrained_no_processing`, so the original weight coordinates (no LayerNorm folding, no centring). float32. |
| Nodes V | The 144 attention heads (layer.head, 12 × 12). |
| Intervention coordinates | a_h(x) = the head output z_h (hook `blocks.L.attn.hook_z`, head h), in R^{T×64}. |
| Edge construction | Definition 1 of the paper, EC_ab(𝒳) = E_{x∼𝒳} ‖J_ba(x)‖²_F, with the **direct-path** block Jacobian. A perturbation δ of z_a enters the residual stream as δ W_O^a and reaches head b only through b's input residual stream, with all other components frozen (the infinitesimal analogue of path patching). So J_ba = ∂z_b/∂resid_pre_{L_b} · W_O^a, taken over all positions, including attention-pattern dependence and the input LayerNorm of block L_b. Defined only for L_a < L_b. |
| Estimator | Hutchinson on the output side: ‖J_b M_a‖²_F = E_u ‖M_aᵀ J_bᵀ u‖² with u ∼ N(0, I) in z_b space. One vector-Jacobian product per (b, probe) gives the estimate for all upstream a. 16 probes per prompt. |
| Prompt distribution 𝒳 | Indirect-object identification (IOI) prompts from 15 fixed templates in the style of Wang et al. (2023), half ABBA and half BABA. All names, places and objects are single GPT-2 tokens. 300 prompts for edge construction (seed 0) and a disjoint set of 300 held-out prompts (seed 1). Disjointness means no identical (template, names, place, object) tuple. |
| Aggregation | The mean over the 300 edge prompts of the per-prompt Hutchinson estimate. Positions are included through the Frobenius norm (the sum over positions). |
| Projection | None. EC is nonnegative. |
| Directionality | Directed, a → b with L_a < L_b. No symmetrisation. |
| Thresholding | Binary graph keeping the top 10% of the 9504 admissible directed pairs (950 edges). Density 5% and 20% are reported as sensitivity only and are not used for the verdict. |
| Null model | 1000 graphs, each from 10 × |E| directed double-edge swaps (a→b, c→d) to (a→d, c→b), accepted only if L_a = L_c and L_b = L_d (layer pair kept), with no duplicate edges. This keeps the in- and out-degree of every node, the edge count of every layer pair, and layer order. Seed 3. |
| Behavioural target | Logit difference LD = logit(IO) − logit(S) at the final position, on the held-out prompts. |

## H1 details

- Degree = in-degree + out-degree in the binary graph.
- k = the 80th percentile of degree over the 144 heads. The rich club R = heads with degree > k.
- Directed rich-club coefficient Φ(k) = E_R / (|R| (|R| − 1)), where E_R is the number of directed edges inside R. Φ_norm(k) = Φ(k) / mean of Φ(k) over the null ensemble. Each null graph is evaluated with its own nodes of degree > k. Degrees are preserved, so R is the same set.
- Importance set I: for each head h, the activation-patching effect on held-out LD. z_h is replaced by its value on the corrupted (ABC) version of the same prompt, where IO and S are replaced by two other names and the template is kept. The effect is the mean absolute change in LD. I = the top |R| heads by effect.
- Overlap = |R ∩ I|. The hypergeometric test uses population 144, |R| successes and |I| draws.

## H2 details

- Communities: Louvain with directed modularity (Leicht and Newman), as in `networkx.community.louvain_communities` on the binary DiGraph, resolution γ = 1. The partition with the highest modularity over 100 seeds (seeds 100 to 199) is kept.
- Labels (Wang et al. 2023, 7 classes, 26 heads):

  | Class | Heads |
  |---|---|
  | name mover | 9.6, 9.9, 10.0 |
  | negative name mover | 10.7, 11.10 |
  | backup name mover | 9.0, 9.7, 10.1, 10.2, 10.6, 10.10, 11.2, 11.9 |
  | S-inhibition | 7.3, 7.9, 8.6, 8.10 |
  | induction (incl. fuzzy) | 5.5, 5.8, 5.9, 6.9 |
  | duplicate token (incl. fuzzy) | 0.1, 0.10, 3.0 |
  | previous token | 2.2, 4.11 |

- AMI (scikit-learn, arithmetic normalisation) between the community labels of these 26 heads and their class labels. Null: 1000 permutations of the class labels over the 26 heads (seed 5).
- Secondary, not used for the verdict: AMI distribution of Louvain partitions (best of 10 seeds) on 200 null graphs.

## Edge validation (a check on the proxy, not on H1 or H2)

- 20 edges: the 10 strongest by EC, plus 10 sampled uniformly from the remaining retained edges (seed 6).
- For each edge, the finite path-patching effect, measured on the held-out prompts: E_x ‖z_b(resid_pre_{L_b} + (z_a^corr − z_a^clean) W_O^a) − z_b(clean)‖²_F, with everything else frozen. This is the finite counterpart of the same direct-path Jacobian edge, using the ABC corruption described above.
- Reported: the Spearman rank correlation between EC and the finite effect over the 20 edges, with a bootstrap 95% CI over prompts. No pass or fail threshold. It is descriptive.

## Uncertainty

- Bootstrap over the 300 edge prompts, 1000 resamples (seed 4). For each resample: recompute EC, threshold, degree, k, R, Φ(k), and the best-of-10 Louvain partition and its AMI. The null mean for Φ_norm is kept from the main run, so null graphs are not re-drawn per resample.
- Reported: 95% percentile CIs for Φ_norm(k), |R ∩ I| and AMI.
- Hutchinson probes: seed 2.

## Reporting commitment

Both verdicts are reported in the paper's worked-example section, whatever they are, together with:
- the full contract table above;
- the null p-values and CIs;
- the density sensitivity;
- the edge-validation correlation;
- every deviation from this file.

A failed prediction is reported as a failed prediction.
