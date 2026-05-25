# Project Handoff

Quick reference for teammates picking up this project. The full implementation plan lives at `.claude/plans/i-have-a-task-floating-neumann.md` (in Adam's local Claude config, not the repo) but everything you need to run things is in this doc + the in-file docstrings.

## What this project does

Research question: **can LLMs faithfully generate stream-of-consciousness text that expresses predefined Big Five personality traits, the way humans do?**

Pipeline:
1. Train a multi-label binary classifier (RoBERTa-base and/or ModernBERT-base) on the Pennebaker & King 1999 Essays dataset to predict Big Five. This is our **probe** — not ground truth, just a calibrated text-personality scorer.
2. Use GPT-4o-mini to generate essays in two regimes:
   - **Style D** — single-trait isolated prompts (HIGH / LOW / NEUTRAL × 5 traits × 100 = 1500 essays). Clean causal design.
   - **Style A** — full multi-trait paired prompts (~500, sampled from human profiles). Realistic multi-trait control.
3. Run the trained probe on the LLM essays. For Style D, compare predicted-probability distributions (humans vs LLM-NEUTRAL vs LLM-HIGH vs LLM-LOW) per trait — Wasserstein-1 distances + bootstrap CIs + KDE plots. For Style A, per-essay alignment metrics.
5. SHAP + LIWC linguistic analysis to compare what cues the probe latches onto in human vs LLM essays.

Paper: ACL-style system description, 8–9 sections, target submission **May 31, 2026**.

## Repo layout

```
.
├── code/
│   ├── datasets/
│   │   ├── essays.csv                            # Pennebaker & King 1999, 2467 essays, cp1252-encoded
│   │   ├── splits/                               # AUTHID-per-line train/val/test splits
│   │   ├── checkpoints/                          # gitignored; ~500MB each
│   │   │   ├── roberta-base_seed42/              # local-only; unzip from Colab
│   │   │   └── ModernBERT-base_seed42/           # local-only; unzip from Colab
│   │   ├── llm_generated/
│   │   │   ├── style_d_single_trait.jsonl        # dry-run: 15 essays (1/trait/level)
│   │   │   └── style_a_paired.jsonl              # dry-run: 5 essays
│   │   └── results/
│   │       ├── dummy/, tfidf-lr/, liwc-lr/       # baseline metrics + per-essay test predictions
│   │       ├── roberta-base/, ModernBERT-base/   # classifier metrics + per-essay test predictions
│   │       └── llm-alignment/                    # phase 4+5 outputs (see below)
│   ├── src/                                      # all the python modules
│   │   ├── config.py                             # paths, trait constants, model defaults
│   │   ├── data.py                               # load essays.csv; produce stratified splits
│   │   ├── baselines.py                          # dummy / TF-IDF+LR / LIWC-style+LR
│   │   ├── classifier.py                         # RoBERTa/ModernBERT trainer + --predict-* CLI
│   │   ├── prompts.py                            # Style D + Style A prompt templates
│   │   ├── generate.py                           # async OpenAI generation with resume + cost guard
│   │   ├── evaluate.py                           # Style D distributional + Style A paired alignment
│   │   └── analyze.py                            # LIWC stats + Style A error dumps + optional SHAP
│   └── notebooks/
│       └── essays.ipynb                          # Colab orchestration: clone, train, push results
├── requirements.txt                              # core deps (install on Colab + local)
├── requirements-dev.txt                          # jupyter/ipykernel (skip on Colab)
├── .env.example                                  # template for the OpenAI key (copy to .env)
├── README.md                                     # initial setup notes (this doc is more current)
└── HANDOFF.md                                    # you are here
```

## Local setup (Windows / macOS / Linux)

Python **3.10–3.12** (3.11 ideal). Adam is on Python 3.10 — newer versions should work but aren't tested.

```bash
git clone https://github.com/avukovic11/evaluating-personality-expression-in-llms.git
cd evaluating-personality-expression-in-llms

# venv
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

# core deps (training, generation, eval)
python -m pip install --upgrade pip
pip install -r requirements.txt

# (optional) dev deps for notebooks
pip install -r requirements-dev.txt
```

**Skip `requirements-dev.txt` on Colab** — Colab provides its own Jupyter stack and our versions conflict with it.

### OpenAI API key

For the generation step (`src/generate.py`) you need an OpenAI API key. The team has $25 of credits in a TakeLab project; ask Adam to add you to it.

Copy `.env.example` to `.env` and paste the key:
```
OPENAI_API_KEY=sk-proj-...
```

`.env` is gitignored. **Never** print the key in a notebook cell that gets committed.

If you have a stale `OPENAI_API_KEY` in your OS environment, our code calls `load_dotenv(override=True)` so the `.env` value wins regardless.

## How to run each piece

All commands run from `code/`:
```bash
cd code
```

### 1. Data (`src/data.py`) — already done

Produces stratified 80/10/10 train/val/test splits (committed in `datasets/splits/`). You only need to rerun if you change the split logic:
```bash
python -m src.data
```

### 2. Baselines (`src/baselines.py`) — already done

Three baselines (Dummy / TF-IDF+LR / LIWC-style+LR). Results in `datasets/results/{dummy,tfidf-lr,liwc-lr}/`:
```bash
python -m src.baselines                  # all three
python -m src.baselines --model tfidf    # just TF-IDF
```

### 3. Classifier (`src/classifier.py`) — RoBERTa results in repo; ModernBERT results in repo; checkpoints local-only

```bash
# train RoBERTa (default)
python -m src.classifier --train --seeds 42

# train ModernBERT
python -m src.classifier --train --model answerdotai/ModernBERT-base --seeds 42

# multi-seed (stretch — currently we only use seed 42)
python -m src.classifier --train --seeds 42,43,44

# quick smoke test (1 epoch on 64 examples, CPU-safe)
python -m src.classifier --train --smoke

# score a single essay (need a checkpoint in datasets/checkpoints/<slug>_seed42/)
python -m src.classifier --predict-file path/to/essay.txt
cat essay.txt | python -m src.classifier --predict-stdin
python -m src.classifier --predict-file essay.txt --model answerdotai/ModernBERT-base
```

**Don't train RoBERTa or ModernBERT locally on CPU** — 4+ hours per epoch. Use Colab; see `code/notebooks/essays.ipynb` for the full Colab recipe. T4 GPU is fine; ~25 min per encoder per seed.

**Checkpoint persistence.** After Colab training, zip the checkpoint dir and download it locally for the demo:
```python
# Colab cell after training:
import shutil
from google.colab import files
shutil.make_archive("/content/roberta-base_seed42", "zip",
                    "datasets/checkpoints/roberta-base_seed42")
files.download("/content/roberta-base_seed42.zip")
```
Then on your local machine, unzip into `code/datasets/checkpoints/roberta-base_seed42/`. The checkpoint is ~500 MB and is gitignored — each teammate fetches it the same way.

### 4. LLM generation (`src/generate.py`) — dry-run done; full run pending

Dry runs (cheap, ~$0.02 total):
```bash
python -m src.generate --style D --n-per-condition 1    # 15 essays
python -m src.generate --style A --n-paired 5           # 5 essays
```

Full runs (~$1.50 total, ~30 min wall clock with concurrency=8):
```bash
python -m src.generate --style D                        # 1500 essays
python -m src.generate --style A                        # 500 essays
```

Useful flags:
- `--model gpt-4o-mini` (default; cheap)
- `--concurrency 8` (in-flight requests)
- `--max-cost 5.0` (auto-abort if estimated USD spend exceeds this)
- `--temperature 0.9`
- The script is **idempotent**: rerunning skips essays already in the output JSONL. Safe to interrupt and resume.

Output JSONL fields (one per essay): `essay_id, prompt_style, prompted_trait, prompted_level, intended_profile, model, finish_reason, prompt_tokens, completion_tokens, latency_s, generated_text`.

### 5. Evaluate (`src/evaluate.py`) — dry-run outputs in repo

Requires a probe checkpoint (see step 3). For Style D, also needs the human-test predictions CSV (from step 3) to anchor the distributional comparison:
```bash
python -m src.evaluate --style D                                              # uses roberta-base by default
python -m src.evaluate --style A
python -m src.evaluate --style D --model answerdotai/ModernBERT-base          # use ModernBERT probe instead
```

Outputs in `datasets/results/llm-alignment/style_{d,a}/<model_slug>/`:
- `metrics.json` — Wasserstein distances + bootstrap CIs (Style D) or per-trait MAE/AUC/acc (Style A)
- `scored_essays.csv` / `predictions.csv` — probabilities per essay (so you can re-analyze without re-running probe inference)
- Style D only: `density_<trait>.png` (KDE plots) and `contamination.png` (5×5 heatmap)

### 6. Analyze (`src/analyze.py`) — LIWC + errors in repo; SHAP pending

Default (LIWC comparison + Style A error dumps; both fast on CPU):
```bash
python -m src.analyze
```

With SHAP (opt-in; slow, prefer GPU):
```bash
python -m src.analyze --shap                                                  # 30 essays per condition
python -m src.analyze --shap --n-shap 10                                      # smaller SHAP sample
```

Outputs in `datasets/results/llm-alignment/analysis/<model_slug>/`:
- `liwc_per_trait.csv` — mean ± std per (trait, condition, feature)
- `liwc_stats_per_trait.csv` — Mann-Whitney U p-values + Cliff's δ for HIGH-vs-LOW, humans-vs-HIGH, humans-vs-LOW
- `errors_per_trait/<trait>.txt` — up to 10 Style A misalignments per trait, full essay text
- `shap_<trait>.csv` (only with `--shap`) — top-K tokens per (trait, condition) by mean |SHAP value|

## What's done

- ✅ Data pipeline + splits
- ✅ All three baselines (Dummy / TF-IDF+LR / LIWC-style+LR) — committed results
- ✅ RoBERTa-base classifier (seed 42, macro-acc 0.585, macro-AUC 0.630) — committed results
- ✅ ModernBERT-base classifier (seed 42) — committed results
- ✅ Prompt templates (Style D + Style A) + smoke-dumped to verify
- ✅ Generation pipeline (async, atomic writes, idempotent resume, cost guard)
- ✅ Dry runs: 15 Style D + 5 Style A essays — committed
- ✅ Evaluation pipeline (distributional + paired) — dry-run outputs committed
- ✅ Linguistic analysis pipeline (LIWC + errors; SHAP opt-in) — dry-run outputs committed

## What's left

- ⏳ **Full LLM generation** — `python -m src.generate --style D` and `python -m src.generate --style A`. Total ~$1.50, ~30 min on a reasonable connection. Run from any machine with the API key set; no GPU needed.
- ⏳ **Full evaluation pass** — re-run `src.evaluate` and `src.analyze` after the full generation. No code changes; just rerun. Use `--shap` for the linguistic analysis (do this on Colab GPU; ~30 min).
- ⏳ **Pick the probe encoder** — compare RoBERTa vs ModernBERT on per-trait AUC; use the stronger one (or both) for the LLM-essay evaluation. See `datasets/results/{roberta-base,ModernBERT-base}/metrics.json`.
- ⏳ **Paper draft** — 8–9 section ACL-style writeup. Target structure documented in the plan.
- ⏳ **Presentation demo** — live demo: take a personality profile from the audience, generate an essay with GPT-4o-mini, score it with the local checkpoint via `python -m src.classifier --predict-stdin`. Requires the checkpoint to be local.

## Important gotchas

1. **`essays.csv` is cp1252-encoded**, not UTF-8 — it has smart quotes from the original Pennebaker dataset. `src/data.py` reads it with `encoding="cp1252"` so this is handled, but if you write a quick standalone script that reads it, remember the encoding.

2. **transformers v5 + fp16 has a bug** — calling `clip_grad_norm_` raises "Attempting to unscale FP16 gradients". `src/classifier.py` uses bf16 on Ampere+ (compute capability ≥ 8) and fp32 on Turing (T4); fp16 is intentionally never used.

3. **Threshold tuning is biased** — the model's predicted probabilities are skewed positive on 4 of 5 traits, but our val-tuned thresholds are mild. We currently report `macro-acc = 0.585` at threshold 0.5; the theoretical ceiling at the same model is `macro-AUC = 0.630`. The paper leads with AUC and the distributional metrics so this calibration miss doesn't bite us. There's a 1–2 pp improvement available from smarter thresholding if needed.

4. **Don't commit the notebook with cell outputs** — `code/notebooks/essays.ipynb` was used to push results from Colab, including a `getpass` step for a GitHub PAT and (separately) the OpenAI API key. **Always `Edit → Clear All Outputs` before saving.** A prior PAT was once printed accidentally and had to be revoked. The current committed notebook has all outputs cleared.

5. **Colab session ephemerality** — model checkpoints live only on the Colab VM and disappear at session end. Either run the full eval/SHAP pipeline in the same session as training, or download the checkpoint zip and unzip locally for re-use.

6. **`generate.py` cost guard is a soft limit** — with concurrency=8, up to 8 already-launched requests will finish after the budget is exceeded. In practice this means it might overshoot by a few cents on a $5 limit.

7. **`generate.py` retries on RateLimitError EXCEPT `insufficient_quota`** — `insufficient_quota` means the project has no credits (billing issue, not transient) and fails fast. If you see this error, you're using a key whose project hasn't been credited.

## Key references

- Pennebaker & King 1999 — original Essays dataset
- Mehta et al. 2020a — BERT-base on Essays, 60.6% macro-acc
- Kerz et al. 2022 — BERT + 437 psycholinguistic features, 63.5% macro-acc (current SOTA on Essays)
- V Ganesan et al. 2023 — zero-shot Big Five with GPT-3 (frames the LLM-as-author angle)
- Naz et al. 2025 — survey of personality detection methods

## Communication

Adam is the primary point of contact. The full plan + decision log lives in `.claude/plans/i-have-a-task-floating-neumann.md` (Adam's local Claude config, not in repo) — ask him to share if you need the rationale for any specific design choice.
