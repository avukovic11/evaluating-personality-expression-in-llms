# Project Handoff

Quick reference for teammates picking up this project. The full implementation plan lives at `.claude/plans/i-have-a-task-floating-neumann.md` (in Adam's local Claude config, not the repo) but everything you need to run things is in this doc + the in-file docstrings.

## What this project does

Research question: **can LLMs faithfully generate stream-of-consciousness text / interview answers that express predefined Big Five personality traits, the way humans do?**

Two tracks running in parallel:

**Track 1 — Essays** (Pennebaker & King 1999 dataset)
1. Train a multi-label binary classifier (RoBERTa-base and/or ModernBERT-base) on 2467 stream-of-consciousness essays to predict Big Five labels. This is our **probe** — not ground truth, just a calibrated text-personality scorer.
2. Use GPT-4o-mini to generate essays in two regimes:
   - **Style B** — single-trait isolated prompts (HIGH / LOW / NEUTRAL × 5 traits × ~127 = 1522 essays total). Clean causal design.
   - **Style A** — full multi-trait paired prompts (500, sampled from human profiles). Realistic multi-trait control.
3. Run the trained probe on the LLM essays. For Style B: compare predicted-probability distributions (humans vs LLM-NEUTRAL vs LLM-HIGH vs LLM-LOW) per trait — Wasserstein-1 distances + bootstrap CIs + KDE plots + contamination heatmap. For Style A: per-trait MAE / AUC / accuracy.
4. LIWC + SHAP linguistic analysis to compare what cues the probe uses in human vs LLM essays.

**Track 2 — RecruitView** (AI4A-lab/RecruitView HuggingFace dataset)
1. Train a regression probe (RoBERTa-base and/or ModernBERT-base) on ~2011 video-interview transcript rows (331 users, ~6 clips each) to predict continuous OCEAN scores. User-level splits prevent data leakage.
2. Use GPT-4o-mini to generate interview answers in two regimes:
   - **Style B** — single-trait isolated (HIGH / LOW / NEUTRAL × 5 traits × 65 questions = 3700 answers).
   - **Style A** — full multi-trait paired (1395 answers, conditioned on human user profiles).
3. Same evaluation pipeline as Track 1, adapted for regression (Spearman/MAE instead of W1/AUC).

Paper: ACL-style system description, 8–9 sections, target submission **May 31, 2026**.

## Repo layout

```
.
├── code/
│   ├── datasets/
│   │   ├── essays.csv                              # Pennebaker & King 1999, 2467 essays, cp1252-encoded
│   │   ├── splits/                                 # AUTHID-per-line train/val/test splits (essays)
│   │   ├── splits_recruitview/                     # user_no-per-line splits (recruitview, user-level)
│   │   ├── checkpoints/                            # gitignored; ~500MB–2GB each
│   │   │   ├── roberta-base_seed42/                # essays probe (RoBERTa)
│   │   │   ├── ModernBERT-base_seed42/             # essays probe (ModernBERT)
│   │   │   ├── roberta-base_recruitview_seed42/    # recruitview probe (RoBERTa)
│   │   │   └── ModernBERT-base_recruitview_seed42/ # recruitview probe (ModernBERT)
│   │   ├── llm_generated/
│   │   │   ├── style_b_single_trait.jsonl          # FULL RUN: 1522 essays
│   │   │   └── style_a_paired.jsonl                # FULL RUN: 500 essays
│   │   ├── llm_generated_recruitview/
│   │   │   ├── style_b_recruitview.jsonl           # FULL RUN: 3700 answers
│   │   │   └── style_a_recruitview.jsonl           # FULL RUN: 1395 answers
│   │   └── results/
│   │       ├── dummy/, tfidf-lr/, liwc-lr/         # essays baselines
│   │       ├── dummy_recruitview/, tfidf-ridge_recruitview/, liwc-ridge_recruitview/
│   │       ├── roberta-base/, ModernBERT-base/     # essays probe metrics + test predictions
│   │       ├── roberta-base_recruitview/, ModernBERT-base_recruitview/
│   │       └── llm-alignment/                      # evaluate.py + analyze.py outputs
│   │           ├── style_b/roberta-base/           # essays Style B distributional metrics
│   │           ├── style_a/roberta-base/           # essays Style A alignment metrics
│   │           ├── analysis/roberta-base/          # essays LIWC + SHAP
│   │           └── recruitview/                    # same structure for Track 2 (after rerun)
│   ├── src/
│   │   ├── config.py           # paths, trait constants, model defaults (both tracks)
│   │   ├── data.py             # load essays.csv; stratified AUTHID splits
│   │   ├── data_recruitview.py # load HuggingFace RecruitView; user-level splits
│   │   ├── baselines.py        # dummy / TF-IDF+LR / LIWC-style+LR (essays)
│   │   ├── classifier.py       # RoBERTa/ModernBERT trainer + --predict-* CLI (both tracks)
│   │   ├── prompts.py          # Style B + Style A prompt templates (both tracks)
│   │   ├── generate.py         # async OpenAI generation with resume + cost guard
│   │   ├── evaluate.py         # distributional + paired alignment (--dataset essays|recruitview)
│   │   └── analyze.py          # LIWC + Style A errors + optional SHAP (--dataset essays|recruitview)
│   └── notebooks/
│       └── essays.ipynb        # Colab orchestration: clone, train, push results
├── requirements.txt
├── requirements-dev.txt
├── .env.example
├── README.md
└── HANDOFF.md                  # you are here
```

## Local setup

Python **3.10–3.12** (3.11 ideal).

```bash
git clone https://github.com/avukovic11/evaluating-personality-expression-in-llms.git
cd evaluating-personality-expression-in-llms

python -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .\.venv\Scripts\Activate.ps1    # Windows PowerShell

pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt   # Jupyter; skip on Colab
```

**Checkpoints** are gitignored (~500 MB–2 GB each). Colleague provides them as zip files; unzip each into `code/datasets/checkpoints/<name>/` (the zip should unpack to files directly in that folder, not a nested subdir). Four checkpoints:
- `roberta-base_seed42/` — essays probe
- `ModernBERT-base_seed42/` — essays probe
- `roberta-base_recruitview_seed42/` — recruitview probe
- `ModernBERT-base_recruitview_seed42/` — recruitview probe

### OpenAI API key

For `src/generate.py` only. Copy `.env.example` → `.env` and fill in the key. `.env` is gitignored; never print it in a notebook cell that gets committed.

## How to run each piece

All commands from `code/`:
```bash
cd code
```

### 1. Data (`src/data.py` / `src/data_recruitview.py`) — already done

Splits are committed. Only rerun if split logic changes.
```bash
python -m src.data
python -m src.data_recruitview
```

### 2. Baselines (`src/baselines.py`) — already done

```bash
python -m src.baselines            # essays (dummy / tfidf-lr / liwc-lr)
# recruitview baselines run separately; results already in results/
```

### 3. Classifier (`src/classifier.py`) — results committed; checkpoints local-only

```bash
# Train (GPU recommended — T4 ~25 min per encoder)
python -m src.classifier --train --seeds 42
python -m src.classifier --train --model answerdotai/ModernBERT-base --seeds 42
python -m src.classifier --train --dataset recruitview --seeds 42
python -m src.classifier --train --dataset recruitview --model answerdotai/ModernBERT-base --seeds 42

# Score a single essay (requires local checkpoint)
python -m src.classifier --predict-file path/to/essay.txt
cat essay.txt | python -m src.classifier --predict-stdin
```

### 4. LLM generation (`src/generate.py`) — FULL RUNS DONE

Full generation is complete. Only rerun to extend or re-generate.

```bash
# Dry run (cheap, just to test):
python -m src.generate --style B --n-per-condition 1
python -m src.generate --style A --n-paired 5
python -m src.generate --dataset recruitview --style B --n-per-condition 1

# Full run (already done; don't rerun unless regenerating):
python -m src.generate --style B          # 1522 essays, ~$0.75
python -m src.generate --style A          # 500 essays, ~$0.50
python -m src.generate --dataset recruitview --style B     # 3700 answers
python -m src.generate --dataset recruitview --style A     # 1395 answers
```

The script is **idempotent**: rerunning skips already-generated entries.

### 5. Evaluate (`src/evaluate.py`) — rerun needed with full data

Requires local checkpoints (see above). Run from `code/`:

```bash
# Track 1 — Essays
python -m src.evaluate --style B                                          # W1 + KDE + contamination heatmap
python -m src.evaluate --style A                                          # per-trait MAE/AUC/acc
python -m src.evaluate --style B --model answerdotai/ModernBERT-base     # ModernBERT comparison

# Track 2 — RecruitView
python -m src.evaluate --dataset recruitview --style B
python -m src.evaluate --dataset recruitview --style A
```

Outputs in `datasets/results/llm-alignment/{style_b,style_a}/<model_slug>/`:
- `metrics.json` — Wasserstein distances + CIs (Style B) or per-trait MAE/AUC/acc (Style A)
- `scored_essays.csv` — probabilities per essay (for re-analysis without re-inference)
- Style B only: `density_<trait>.png`, `contamination.png`

RecruitView outputs in `datasets/results/llm-alignment/recruitview/{style_b,style_a}/<model_slug>/`.

### 6. Analyze (`src/analyze.py`) — rerun needed with full data

```bash
# Track 1 — Essays (LIWC + error dumps; fast on CPU)
python -m src.analyze

# Track 2 — RecruitView
python -m src.analyze --dataset recruitview

# SHAP (slow; use Colab GPU)
python -m src.analyze --shap
python -m src.analyze --dataset recruitview --shap
```

Outputs in `datasets/results/llm-alignment/analysis/<model_slug>/`:
- `liwc_per_trait.csv` — mean ± std per (trait, condition, feature)
- `liwc_stats_per_trait.csv` — Mann-Whitney U p-values + Cliff's δ
- `errors_per_trait/<trait>.txt` — Style A misalignments per trait
- `shap_<trait>.csv` (with `--shap`) — top-K tokens per (trait, condition)

## Key results (full run, 2026-05-29)

### Essays — Style B: W1 HIGH↔LOW (larger = LLM steered the trait further)

| Trait | RoBERTa | ModernBERT |
|---|---|---|
| cOPN | **0.303** | 0.213 |
| cNEU | 0.196 | **0.432** |
| cAGR | 0.197 | **0.382** |
| cCON | 0.107 | 0.077 |
| cEXT | 0.114 | 0.172 |

- **Openness** most steerable by RoBERTa (W1=0.303, HIGH mean prob=0.853).
- **Neuroticism and Agreeableness** much more separated under ModernBERT (W1≈0.43/0.38 vs 0.20).
- **Conscientiousness** weakest on both — LOW essays still score high; GPT's default writing is already conscientious-sounding.
- GPT NEUTRAL writing scores higher than human baseline on cCON and cAGR (GPT default style = polished + agreeable).
- Contamination: prompting HIGH-NEU strongly suppresses probe's cOPN score (−0.192 in RoBERTa heatmap) — the two traits are linguistically entangled.

### Essays — Style A: macro alignment (n=500 multi-trait essays)

| Metric | RoBERTa | ModernBERT |
|---|---|---|
| AUC | **0.819** | 0.774 |
| Accuracy | **0.646** | 0.644 |
| Profile exact match | 0.15 | — |

Per-trait AUC (RoBERTa): cNEU 0.871, cOPN 0.851, cAGR 0.810, cEXT 0.806, cCON 0.755.
**RoBERTa is the stronger primary probe for essays.**

### RecruitView — Style B: W1 HIGH↔LOW

| Trait | RoBERTa | ModernBERT |
|---|---|---|
| openness | 0.070 | 0.061 |
| conscientiousness | 0.058 | **0.140** |
| extraversion | 0.030 | 0.077 |
| agreeableness | 0.047 | 0.060 |
| neuroticism | 0.010 | 0.049 |

W1 values are 3–10× smaller than essays. GPT barely separates HIGH from LOW as seen by either probe. All GPT-generated interview answers cluster at **negative z-scores** (e.g., openness HIGH=−0.53, LOW=−0.46) while human test mean ≈ +0.07 — severe domain gap.

### RecruitView — Style A: macro Spearman

| | RoBERTa | ModernBERT |
|---|---|---|
| per-essay | −0.216 | −0.157 |
| per-user | −0.480 | −0.392 |

**Negative correlations on both probes** — not a probe artefact. GPT's uniformly polished answers look like low-trait text to both models. See gotcha #9 for paper framing.

---

## What's done

- ✅ Data pipeline + splits (essays + recruitview)
- ✅ All baselines (essays: dummy/tfidf-lr/liwc-lr; recruitview: dummy/tfidf-ridge/liwc-ridge) — committed results
- ✅ RoBERTa-base probe — essays: macro-acc 0.584, macro-AUC **0.630**; recruitview: macro-Spearman 0.269 per-answer / 0.472 user-aggregated
- ✅ ModernBERT-base probe — essays: macro-acc 0.568, macro-AUC **0.620**; recruitview: macro-Spearman 0.301 / 0.470 user-aggregated
- ✅ RoBERTa is the stronger essays probe; performance is close on recruitview
- ✅ Prompt templates (Style B + Style A for both tracks) + smoke-tested
- ✅ Generation pipeline (async, idempotent, cost guard)
- ✅ **Full LLM generation (essays)**: 1522 Style B + 500 Style A — committed
- ✅ **Full LLM generation (recruitview)**: 3700 Style B + 1395 Style A — committed
- ✅ All 4 checkpoints unzipped locally (required to run evaluate + analyze)
- ✅ **Full evaluation pass (both probes, both tracks)** — results committed; see Key Results section above
- ✅ Linguistic analysis pipeline (LIWC + errors + optional SHAP, both tracks) — dry-run outputs committed; **full rerun still needed**

## What's left

- ⏳ **Full evaluate.py pass** — running now: `src.evaluate --style B`, `--style A` (essays) and same with `--dataset recruitview`. Produces the KDE/heatmap plots and W1 metrics that feed the paper's results section.
- ⏳ **Full analyze.py pass** — once evaluate.py finishes (needs Style A `predictions.csv`): `src.analyze` and `src.analyze --dataset recruitview`.
- ⏳ **SHAP analysis** — `src.analyze --shap` (essays + recruitview). GPU strongly recommended; use Colab T4. ~30–60 min each. Notebook: `code/notebooks/essays.ipynb` (adapt for Colab run).
- ⏳ **Choose primary probe** — RoBERTa leads essays AUC (0.630 vs 0.620). Plan: report both, call RoBERTa primary. Recruitview is effectively tied.
- ⏳ **Paper draft** — 8–9 section ACL-style writeup. Results section depends on the full evaluate + analyze outputs.
- ⏳ **Presentation demo** — live demo: audience gives a personality profile → GPT-4o-mini generates an essay → `python -m src.classifier --predict-stdin` scores it. Requires the checkpoint to be local.
- ⏳ **analyze.py extensions** *(code work; redo after full generation)*:
  - Add `run_tfidf_comparison()` — discriminating TF-IDF tokens (humans vs HIGH vs LOW)
  - Add `run_keyword_frequency()` — trait-keyword echo rate + Mann-Whitney + Cliff's δ
  - Add `TRAIT_KEYWORDS` to `config.py` — flat word lists for analysis
  - Improve `_aggregate_shap()` — add signed SHAP column + `min_count=3` doc filter
  - Add `--humans-only` mode — compares human_high vs human_low using ground-truth labels
  - Add `--plot` flag — generates LIWC heatmap + TF-IDF diverging bar chart PNGs

## Important gotchas

1. **`essays.csv` is cp1252-encoded** (smart quotes). `src/data.py` handles it; remember if writing standalone scripts.

2. **transformers v5 + fp16 bug** — `clip_grad_norm_` raises an error. `src/classifier.py` uses bf16 on Ampere+ and fp32 on T4; fp16 is never used.

3. **Threshold tuning is biased** — predicted probabilities are skewed positive on 4 of 5 traits. Paper leads with AUC and distributional metrics to avoid this biting us.

4. **Don't commit notebooks with cell outputs** — `essays.ipynb` has a `getpass` step for GitHub PAT and OpenAI key. Always `Edit → Clear All Outputs` before saving. A prior PAT was accidentally printed and had to be revoked.

5. **Checkpoint path structure** — each checkpoint dir must contain `config.json`, `model.safetensors`, `tokenizer.json` etc. directly (not nested in a subdirectory). Zips from Colab sometimes double-nest; flatten with `mv inner/* outer/ && rmdir inner` if needed.

6. **RecruitView probe outputs z-scores**, not sigmoid probabilities. `evaluate.py` and `analyze.py` handle this automatically when `--dataset recruitview` is passed.

7. **Style B recruitview uses lowercase trait names** (`openness`, `conscientiousness`, …) in the JSONL, not the `cEXT`-style column names. `evaluate.py` maps these at load time.

8. **`generate.py` retries on RateLimitError EXCEPT `insufficient_quota`** — `insufficient_quota` means billing issue; fails fast.

9. **RecruitView Style A shows negative Spearman correlations** (−0.09 to −0.28 per-essay, −0.25 to −0.59 per-user) — this is not a bug. The probe was trained on real human video transcripts (natural, varied, imperfect speech). GPT generates uniformly clean, structured answers regardless of the intended personality level, so the probe scores all GPT output as below-average. The negative correlation arises because the more extreme a human's real trait score (high intended z), the more GPT's generic answer diverges from that authentic style. Frame this as a **domain transfer failure**, not just a steering failure: W1 HIGH↔LOW within GPT-generated text is near-zero (0.010–0.070), confirming the probe can barely distinguish GPT-HIGH from GPT-LOW at all. Contrast with essays, where W1 ranges 0.107–0.303.

## Key references

- Pennebaker & King 1999 — original Essays dataset
- AI4A-lab/RecruitView — HuggingFace video interview dataset with OCEAN labels
- Mehta et al. 2020a — BERT-base on Essays, 60.6% macro-acc
- Kerz et al. 2022 — BERT + 437 psycholinguistic features, 63.5% macro-acc (SOTA on Essays)
- V Ganesan et al. 2023 — zero-shot Big Five with GPT-3
- Naz et al. 2025 — survey of personality detection methods

## Communication

Adam is the primary contact. Full plan + decision log: `.claude/plans/i-have-a-task-floating-neumann.md` (Adam's local Claude config, not in repo).
