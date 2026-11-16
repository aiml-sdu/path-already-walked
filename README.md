# A Path Already Walked: worked example

Code, pre-registration and results for the worked example in

> Phongsakon Mark Konrad, Toygar Tanyel, Serkan Ayvaz. *A Path Already Walked: On Inheriting Network-Neuroscience Tools for Mechanistic Interpretability.* arXiv, 2026.

The paper argues that a graph claim about a transformer is only meaningful once it declares its nodes, edges, prompt distribution, threshold, null model and behavioural target. This repository runs one such declared test on GPT-2 small. It builds the paper's gradient effective-connectivity graph over the 144 attention heads on indirect-object-identification prompts. It then tests two translations from the paper's Table 2, rich-club ordering and modularity, against a null that keeps degree, layer-pair edge counts and layer order. Both predictions were pre-registered, and both fail. The paper (Section 6) and `RESULTS.md` explain why.

## Pre-registration

`PREREG.md` fixes every choice and quotes the failure criteria verbatim from the paper. It was written before any statistic was computed. Its SHA-256 is stored in `PREREG.sha256`, the script records it in `results/results.json`, and you can check it with

```bash
shasum -a 256 PREREG.md
```

## Install

Python 3.12. The pinned versions are in `requirements.txt` (torch 2.15.0, transformer-lens 3.9.0). TransformerLens 4 removed the `HookedTransformer` API used here, so keep the pin.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

The GPT-2 small weights (`gpt2`, about 550 MB) are downloaded from Hugging Face on first use.

## Run

```bash
.venv/bin/python poc.py --smoke   # 8 prompts, shapes and timings only, no test statistics
.venv/bin/python poc.py           # the pre-registered run, writes results/
```

The full run takes about 9 minutes on an Apple M3 Pro (MPS). CPU also works and is slower. Edge estimates on MPS and CPU agree to a relative difference of 4e-6. The script also checks that its hand-written attention layer reproduces the model's own head outputs.

## What is in the repository

| File | Contents |
|---|---|
| `poc.py` | The whole experiment: prompts, edges, held-out patching, nulls, both tests, bootstrap, edge validation |
| `PREREG.md`, `PREREG.sha256` | The pre-registration and its hash |
| `RESULTS.md` | The verdicts, sensitivity, edge validation, and a post-hoc diagnosis labelled as such |
| `results/results.json` | Every reported statistic, with configuration and seeds |
| `results/ec_mean.npy` | The 144 × 144 mean effective-connectivity matrix (sender, receiver) |

The per-prompt matrix used for the bootstrap (`ec_per_prompt.npy`, 24 MB) is not committed. The run regenerates it.

## Paper figures and tables

| Paper item | Produced by |
|---|---|
| Table 4 (the contract) | `PREREG.md` |
| Section 6, all numbers | `poc.py` → `results/results.json` |

## Citation

```bibtex
@misc{konrad2026pathalreadywalked,
  title         = {A Path Already Walked: On Inheriting Network-Neuroscience Tools for Mechanistic Interpretability},
  author        = {Konrad, Phongsakon Mark and Tanyel, Toygar and Ayvaz, Serkan},
  year          = {2026},
  eprint        = {TBA},
  archivePrefix = {arXiv},
  note          = {An earlier version was presented at the ICML 2026 Workshop on Mechanistic Interpretability}
}
```

The arXiv identifier will be added once it is assigned.

## Licence

MIT, see `LICENSE`.
