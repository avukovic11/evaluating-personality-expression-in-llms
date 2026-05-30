# Phase 5 — analyze.py extensions (handoff)

Continuing from a prior chat. Read this + `HANDOFF.md` in the repo and you're caught up.

## Current state (2026-05-26)

- 5 days to deadline (May 31, 2026).
- Full LLM generation (1500 Style D + 500 Style A) is being kicked off by a teammate — ~30 min wall clock, ~$1.50 budget. All Phase 5 work below can run in parallel with that.
- `src/analyze.py` already implements LIWC + Style A error dumps + SHAP, all in the correct three-way shape (humans / Style-D-HIGH / Style-D-LOW per trait). The new analyses follow the same pattern.
- `src/evaluate.py` already produces the Style D Wasserstein-1 metrics, KDE plots, and cross-trait contamination heatmap. Style A produces per-trait MAE / AUC / accuracy + profile exact-match.

## Design decisions reached in prior chat

- **RQ3 = comparison.** All linguistic analyses must compare humans vs LLM. The existing `analyze.py` is already comparative; new analyses adopt the same shape.
- **cNEU=1 = high neuroticism** (anxious / worried / stressed pole) for both labeled humans and prompted LLM. Confirmed; no sign flip needed.
- **Style A primary metrics: per-trait MAE + per-trait AUC across 500 essays.** These are interpretable, well-estimated, and directly comparable to Style D per-trait results. Profile exact-match is fine as a one-line headline only. Don't use per-essay 5-trait correlation — Pearson on n=5 is too noisy and undefined on all-HIGH / all-LOW intended profiles.
- **Contamination heatmap = main result, not stretch.** Data falls out of Style D for free. Elevates the paper from "did GPT shift each trait" to "are the trait dimensions actually independent."
- **Two confound checks to add:** word count per (trait × level) cell, and trait-keyword echo rate (HIGH-prompted essays mentioning HIGH-pole words vs LOW-prompted essays mentioning LOW-pole words).

## Paper framing (notes for writeup)

- **W1 distance reports the LLM's prompt-conditioned shift *as perceived by this probe*.** Per-trait AUC ranges 0.572–0.711. Small W1 on Conscientiousness (AUC 0.572) or Agreeableness (AUC 0.591) is genuinely ambiguous — it could mean GPT can't steer those traits OR the probe is partly blind to them in human writing too. Paper text must say this explicitly, not just in Limitations.
- **NEUTRAL = GPT's default *scored by a probe trained on human stream-of-consciousness*.** Domain-transfer caveat: NEUTRAL probabilities being off-center could reflect register/structural differences (more polished, less rambly) rather than personality. Mention directly in Discussion, not only Limitations.
- **"HIGH human" (labeled cTRAIT=1) ≠ "HIGH LLM" (prompted HIGH).** Observed vs intended categories. One Methods sentence acknowledging the asymmetry.

## Tasks in order

1. **Define `TRAIT_KEYWORDS` in `src/config.py`.** Single source of truth for both LLM prompt vocabulary and the keyword-frequency analysis. Format:
   ```python
   TRAIT_KEYWORDS = {
       "cEXT": {"high": [...], "low": [...]},
       "cNEU": {"high": ["anxious", "worried", "stressed", "tense", "nervous"],
                "low":  ["calm", "stable", "relaxed", "secure", "even-tempered"]},
       # cAGR, cCON, cOPN
   }
   ```
   Pick ~5–7 words per pole, plain language, roughly matched in formality between HIGH and LOW within a trait.

2. **Refactor `src/prompts.py`** to read from `TRAIT_KEYWORDS`. Currently it probably has its own local definition — extract it. This is what makes the lexical-echo check meaningful: GPT is being prompted with exactly the words we then check for in the output.

3. **Add `run_tfidf_comparison()` to `analyze.py`.** Follows the LIWC pattern. For each trait, fit `TfidfVectorizer` over the corpus (humans + Style-D), compute mean TF-IDF weight per token within each of the three groups (humans / HIGH / LOW), report top-K discriminating tokens per group. Output: `tfidf_per_trait.csv`. CPU, fast (seconds per trait).

4. **Add `run_keyword_frequency()` to `analyze.py`.** For each essay, count pole-keyword occurrences (lowercase, word-boundary regex) using `TRAIT_KEYWORDS`. Aggregate as rate per 1000 tokens, grouped by condition. Stats: Mann-Whitney + Cliff's δ on humans-vs-HIGH using HIGH keywords; humans-vs-LOW using LOW keywords. Output: `keyword_freq_per_trait.csv`. CPU, fast.

5. **Improve `_aggregate_shap`:**
   - Add mean **signed** SHAP as a second column alongside mean |SHAP|. Positive = pulls predictions toward HIGH-on-T, negative = pulls toward LOW. Two-line change. Critical for the "are humans and LLMs signaling HIGH/LOW with the same vocabulary?" comparison.
   - Add a `min_count` filter (default 3) — drop tokens appearing in fewer than 3 essays before sorting, to prevent rare-token dominance of the top-K.

6. **Two cheap additions to `evaluate.py`** (if not already present):
   - Word count per (trait, level) cell as a summary table. Length confound check.
   - Trait-keyword echo rate in LLM essays — overlap with task 4 but specifically as a confound table in the evaluation output, not buried in the analysis dir.

7. **Sampling determinism.** Use `seed=42` everywhere. Existing `run_shap` uses `np.random.default_rng(config.SEED)` — good. For new functions, match. Optionally save sampled AUTHIDs to `datasets/splits/shap_humans/<trait>.txt` for cross-machine reproducibility.

## After full LLM generation finishes

```bash
cd code
python -m src.evaluate --style D          # W1 + KDE + contamination heatmap
python -m src.evaluate --style A          # per-trait MAE/AUC + profile match
python -m src.analyze                     # LIWC + errors + tfidf + keyword_freq (CPU)
python -m src.analyze --shap              # GPU, on Colab; ~1 hr
```

## Workflow

- **Local:** all code edits + everything except SHAP. Probe inference on a few thousand essays is fast on CPU.
- **Colab:** SHAP only. Push code to GitHub, `git pull` in Colab, run, push results.
- **Checkpoint:** download `roberta-base_seed42/` zip from Colab once, unzip to `code/datasets/checkpoints/`. Needed for local `--predict-*` and for the live demo.

## Open question

- Human sample for SHAP: stick with test set (n=248, plan default) or use full train+val+test (n=2467)? Test is more defensible — anchors to the same data the AUC numbers are computed on. Current code uses test (via `load_splits()`). Keep as-is unless there's a specific reason to change.

## Gotchas

- `essays.csv` is **cp1252-encoded** (smart quotes). Handled in `data.py`; remember if writing standalone scripts.
- `transformers` v5 + fp16 has the `clip_grad_norm_` unscale bug → fp32 on T4, bf16 on A100, never fp16. Existing `classifier.get_device()` handles this; `analyze.py` reuses it.
- **Never commit notebook with cell outputs.** A PAT was once printed and had to be revoked.
- SHAP can OOM on long essays. If problems: truncate to 400 tokens or set batch_size=1.
- `generate.py` retries on `RateLimitError` except `insufficient_quota` (billing, fails fast).

## Key files to look at first

- `code/src/analyze.py` — extend with `run_tfidf_comparison` and `run_keyword_frequency`, modify `_aggregate_shap`.
- `code/src/config.py` — add `TRAIT_KEYWORDS`.
- `code/src/prompts.py` — refactor to import `TRAIT_KEYWORDS` from config.
- `code/src/evaluate.py` — optionally add word-count and keyword-echo confound tables.
- `code/src/baselines.py` — has `liwc_features()` already reused by analyze; pattern to follow.
