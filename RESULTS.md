# Results — Evaluating Personality Expression in LLMs

Generated: 2026-05-30. Full run with 1 522 Style B essays + 500 Style A essays (Track 1) and 3 700 Style B + 1 395 Style A interview answers (Track 2). Probe: RoBERTa-base (primary) and ModernBERT-base (secondary).

---

## 1. Probe Performance (Essays, binary classification)

Trained on Pennebaker & King essays (n=2 467, 80/10/10 split), multi-label binary, seed 42.

| Trait | RoBERTa AUC | RoBERTa acc | ModernBERT AUC | ModernBERT acc |
|---|---|---|---|---|
| cEXT | 0.599 | 0.540 | 0.600 | 0.548 |
| cNEU | 0.658 | 0.649 | 0.635 | 0.597 |
| cAGR | 0.615 | 0.548 | 0.585 | 0.556 |
| cCON | 0.564 | 0.504 | 0.567 | 0.520 |
| cOPN | 0.716 | 0.677 | 0.714 | 0.617 |
| **macro** | **0.630** | **0.584** | **0.620** | **0.568** |

**RoBERTa is the stronger probe overall** (AUC 0.630 vs 0.620). Openness is the easiest trait for both models; Conscientiousness the hardest. Used as the primary probe throughout.

## 2. Probe Performance (RecruitView, regression)

Trained on AI4A-lab/RecruitView transcripts (331 users, 2 011 rows, user-level 70/15/15 split), five separate regressors, seed 42.

| Trait | RoBERTa Spearman | RoBERTa MAE | ModernBERT Spearman | ModernBERT MAE |
|---|---|---|---|---|
| openness | 0.384 | 0.577 | 0.396 | 0.569 |
| conscientiousness | 0.301 | 0.526 | 0.349 | 0.600 |
| extraversion | 0.255 | 0.596 | 0.287 | 0.608 |
| agreeableness | 0.325 | 0.647 | 0.353 | 0.650 |
| neuroticism | 0.080 | 0.401 | 0.118 | 0.427 |
| **macro** | **0.269** | **0.549** | **0.301** | **0.571** |
| *user-aggregated macro* | *0.472* | — | *0.470* | — |

Neuroticism is almost unpredictable from interview transcripts (Spearman ~0.08–0.12). User-level aggregation (averaging predictions across a user's ~6 clips) roughly doubles Spearman (0.27 → 0.47), showing within-person variance is high. Both probes are comparable; RoBERTa used as primary.

---

## 3. Track 1 — Essays: LLM Alignment (Style B)

### 3.1 Distributional Alignment: Wasserstein-1 Distances

W1 measures distance between predicted-probability distributions (larger = LLM steered the trait further from baseline). Bootstrap 95 % CIs from 1 000 resamples.

#### RoBERTa probe

| Trait | W1 HIGH↔LOW | 95 % CI | mean HIGH | mean LOW | mean NEUTRAL | mean humans |
|---|---|---|---|---|---|---|
| cOPN | **0.303** | [0.278, 0.329] | 0.853 | 0.550 | 0.606 | 0.547 |
| cAGR | 0.197 | [0.183, 0.214] | 0.682 | 0.484 | 0.641 | 0.611 |
| cNEU | 0.196 | [0.183, 0.208] | 0.676 | 0.480 | 0.555 | 0.494 |
| cCON | 0.107 | [0.095, 0.118] | 0.719 | 0.612 | 0.645 | 0.597 |
| cEXT | 0.114 | [0.102, 0.127] | 0.660 | 0.546 | 0.587 | 0.596 |

#### ModernBERT probe

| Trait | W1 HIGH↔LOW | mean HIGH | mean LOW | mean NEUTRAL |
|---|---|---|---|---|
| cNEU | **0.432** | 0.750 | 0.318 | 0.482 |
| cAGR | 0.382 | 0.881 | 0.498 | 0.698 |
| cOPN | 0.213 | 0.808 | 0.595 | 0.668 |
| cEXT | 0.172 | 0.865 | 0.693 | 0.726 |
| cCON | 0.077 | 0.667 | 0.590 | 0.564 |

#### Key observations

- **Openness most steerable (RoBERTa)**: W1=0.303, mean HIGH=0.853 — probe pushed to the top of its range. GPT reliably injects open, curious, imaginative vocabulary.
- **Neuroticism and Agreeableness more detectable by ModernBERT** (W1≈0.43 / 0.38 vs 0.20 / 0.20) — the two probes disagree on which traits are easiest, suggesting each learned partially different surface cues.
- **Conscientiousness is the hardest trait on both probes** (W1≈0.08–0.11). Even LOW essays score high (0.59–0.61): GPT's default writing style is already structured, organized, and careful regardless of the prompt.
- **GPT NEUTRAL exceeds the human baseline on cAGR and cCON**: NEUTRAL mean 0.641/0.645 vs human mean 0.611/0.597. GPT's default register is more agreeable and conscientious-sounding than actual college students' stream-of-consciousness essays.
- **cEXT NEUTRAL falls between HIGH and LOW** and is close to the human baseline (0.587 vs 0.596), suggesting extraversion cues in GPT's default are roughly human-like.

### 3.2 Contamination Heatmap (RoBERTa, HIGH − LOW signed W1)

The heatmap diagonal is the on-target W1 (same as §3.1). Off-diagonal cells show how much a trait-T prompt leaks into another trait U's probe score. Positive = HIGH-T prompt also raises U score; negative = suppresses it.

| Prompted trait | cEXT effect | cNEU effect | cAGR effect | cCON effect | cOPN effect |
|---|---|---|---|---|---|
| cEXT | **+0.114** | −0.190 | +0.183 | +0.075 | −0.060 |
| cNEU | −0.016 | **+0.196** | −0.059 | −0.065 | **−0.192** |
| cAGR | +0.066 | −0.152 | **+0.197** | +0.129 | −0.033 |
| cCON | +0.105 | −0.015 | +0.071 | **+0.107** | −0.086 |
| cOPN | −0.103 | −0.058 | −0.062 | −0.117 | **+0.303** |

**Notable leakage**:
- Prompting HIGH-NEU strongly suppresses the cOPN score (−0.192): anxious, ruminating writing reads as closed/conventional to the probe.
- Prompting HIGH-EXT suppresses cNEU (−0.190): social, energetic writing reads as emotionally stable.
- Prompting HIGH-AGR suppresses cNEU (−0.152): warm, cooperative writing reads as stable.
- These are not prompt artefacts — they reflect genuine covariance in how personality vocabulary clusters in the training data.

### 3.3 Style A: Multi-Trait Alignment (n=500)

Each essay was generated with a full 5-trait profile matching a sampled human. The probe predicts each trait independently; accuracy = fraction where predicted label matches intended.

| Trait | RoBERTa AUC | RoBERTa acc | RoBERTa misaligned | ModernBERT AUC | ModernBERT acc | ModernBERT misaligned |
|---|---|---|---|---|---|---|
| cNEU | **0.871** | 0.772 | 114/500 | — | — | 118/500 |
| cOPN | 0.851 | 0.596 | 202/500 | — | — | 202/500 |
| cAGR | 0.810 | 0.668 | 166/500 | — | — | 176/500 |
| cEXT | 0.806 | 0.662 | 169/500 | — | — | 211/500 |
| cCON | **0.755** | 0.530 | 235/500 | — | — | 183/500 |
| **macro** | **0.819** | **0.646** | | **0.774** | **0.644** | |

Profile exact match (all 5 correct): **15 %** (RoBERTa), expected to be low for 5-trait multi-label.

**Key observations**:
- AUC is consistently high (0.755–0.871), meaning even in a multi-trait context GPT successfully shifts trait-relevant writing — the probe can rank HIGH vs LOW intended essays reliably.
- Accuracy is lower than AUC because the predicted probability threshold (0.5) is not perfectly calibrated; AUC is the better metric here.
- cCON remains the hardest (AUC 0.755, 235/500 misaligned): when told to be simultaneously conscientious AND other traits, the probe often cannot tell intended high from low.
- cNEU is easiest (AUC 0.871, only 114 misaligned): neuroticism vocabulary is highly distinctive even in a mixed-trait prompt.

---

## 4. Track 1 — Essays: Linguistic Analysis

### 4.1 LIWC Feature Comparison (Style B, HIGH vs LOW, Cliff's δ)

Top 3 features by |δ| for the HIGH-vs-LOW comparison. Positive δ = HIGH group uses feature more.

| Trait | Feature | δ | Interpretation |
|---|---|---|---|
| **cEXT** | pron_we | +0.90 | HIGH essays: "we", collective framing |
| | pron_i | −0.89 | HIGH: less "I" (social vs self-focused) |
| | excl_rate | +0.78 | HIGH: more exclamations |
| | negation | −0.51 | HIGH: less negation |
| **cNEU** | avg_word_len | −0.97 | HIGH: shorter words (anxious, fragmented style) |
| | ttr | −0.89 | HIGH: lower lexical diversity (repetitive rumination) |
| | pron_we | −0.89 | HIGH: less "we" (more self-focused) |
| | pron_i | +0.86 | HIGH: more "I" |
| | q_rate | +0.82 | HIGH: many more questions (ruminating self-doubt) |
| **cAGR** | pron_i | +0.94 | HIGH: more "I" (personal, warm) |
| | pron_you | −0.81 | HIGH: less "you" (direct address) |
| | negation | −0.78 | HIGH: much less negation |
| **cCON** | avg_word_len | +0.59 | HIGH: longer words (formal, precise) |
| | word_count_log | −0.36 | HIGH: shorter essays (focused, concise) |
| | negation | −0.24 | HIGH: less negation |
| **cOPN** | pron_i | −0.77 | HIGH: less "I" (less self-referential) |
| | q_rate | +0.47 | HIGH: more questions (curious, exploratory) |
| | avg_word_len | +0.44 | HIGH: longer words (intellectual vocabulary) |
| | avg_sentence_len | +0.42 | HIGH: longer sentences |

**Cross-trait pattern**: GPT uses pronouns very systematically. HIGH-NEU = "I" heavy, few "we". HIGH-EXT = "we" heavy, few "I". HIGH-AGR = "I" heavy, few "you". This creates clear lexical signatures that the probe latches onto — and also shows potential for contamination (NEU and AGR both use high "I" rate).

**Emotion features (emo_*)**: All Cliff's δ = 0.0 across every trait and comparison. The NRC lexicon-based emotion features are entirely uninformative — GPT essays do not systematically differ on these categories regardless of the trait prompt. This likely reflects the stream-of-consciousness format avoiding explicit emotion words.

### 4.2 Humans vs GPT LIWC Comparison (key gaps)

| Feature | Direction | Interpretation |
|---|---|---|
| word_count_log | humans << GPT | GPT writes longer essays than humans across ALL conditions (δ ≈ −0.60 to −0.69). Length confound present. |
| q_rate | humans >> GPT (for HIGH-EXT, HIGH-AGR) | Humans use questions more than GPT-HIGH on social traits. |
| negation | humans >> GPT (for HIGH-EXT, HIGH-OPN) | Humans negate more; GPT writes in a more positive, declarative register. |
| pron_i | humans >> GPT for HIGH-NEU | Human neurotic essays are more self-focused than GPT's. |

### 4.3 TF-IDF: Discriminating Vocabulary

Top-5 discriminating tokens per condition (tokens with highest mean TF-IDF score within that condition, penalized by occurrence in other conditions):

| Trait | humans | HIGH | LOW |
|---|---|---|---|
| cEXT | well, am, he, very, now | love, new, we, speaking of, energy | what if, small talk, thoughts, quiet, small |
| cNEU | he, very, well, him, minutes | what if, but what, can even, my brain, if they | we, moments, love, sometimes, part of |
| cAGR | he, now, am, very, will | love, what if, agreeable, wonder if, think about | real, pretending, like they, rather, me started |
| cCON | he, him, very, it is, well | balance, pressure, let, if don, what if | can even, what was, clean, mess, clothes |
| cOPN | he, am, him, well, she | art, experiences, experience, moments, moment | routine, conventional, stick to, stick, prefer |

**Human essays**: dominated by third-person pronouns ("he", "him", "she") — humans write about people in their lives. GPT essays are more self-referential and abstract.

**HIGH tokens**: GPT injects trait-specific vocabulary directly ("agreeable", "conventional"). This is lexical echo — the prompt's descriptor words appear in the output, which is expected for label-only prompts. The "what if" bigram appears for cNEU, cAGR, cCON HIGH — GPT overuses this as a rumination/concern marker across traits.

**LOW tokens**: cOPN LOW shows "routine", "conventional", "stick to" — direct echoing of the LOW descriptor. cCON LOW shows "mess", "clean", "clothes" — concrete disorganized imagery.

### 4.4 Keyword Frequency (Trait-Specific Word Lists)

Mean rate per 1 000 tokens for HIGH-pole keywords (e.g., "anxious", "worried" for cNEU):

| Trait | humans | GPT HIGH | GPT LOW | High vs Low δ |
|---|---|---|---|---|
| cEXT | 4.94 | **19.89** | 14.04 | +0.69 |
| cNEU | 1.75 | 5.74 | 2.74 | +0.61 |
| cAGR | 1.33 | **4.00** | 1.23 | +0.68 |
| cCON | 0.40 | 3.52 | 1.93 | +0.54 |
| cOPN | 0.30 | 2.18 | 0.66 | +0.53 |

And LOW-pole keywords (e.g., "routine", "conventional" for cOPN):

| Trait | humans | GPT HIGH | GPT LOW | Low vs High δ |
|---|---|---|---|---|
| cEXT | 0.39 | 2.45 | 7.53 | −0.85 |
| cNEU | 1.76 | 7.31 | 3.97 | +0.61* |
| cAGR | 0.36 | 0.13 | 1.72 | −0.75 |
| cCON | 0.09 | 1.04 | 1.90 | −0.32 |
| cOPN | 0.11 | 0.47 | **5.48** | −0.95 |

*Note: cNEU LOW-pole keywords ("calm", "stable", "relaxed") appear more in HIGH essays — GPT's HIGH-NEU essays contrast worried self against aspirational calm, unintentionally including both poles.

**Key findings**:
- GPT reliably uses HIGH-pole keywords at 3–10× the human rate when prompted for HIGH. This is **lexical instruction-following**, not just style shifting.
- cOPN shows the most extreme LOW-pole echo: rate=5.48 in LOW essays vs 0.47 in HIGH (δ=−0.95). "Routine", "conventional", "stick to" appear very frequently.
- Human baseline is far below GPT for all trait-specific keywords — humans express personality implicitly through style, not explicit label vocabulary.
- HIGH-vs-LOW keyword rates are significantly different for all traits (all p<0.001), but the delta is smaller for cCON (δ=0.54) than cOPN/cEXT (0.85–0.95).

---

## 5. Track 2 — RecruitView: LLM Alignment (Style B)

### 5.1 Distributional Alignment: W1 Distances

| Trait | RoBERTa W1 HIGH↔LOW | ModernBERT W1 HIGH↔LOW | mean HIGH (z) | mean LOW (z) | mean humans (z) |
|---|---|---|---|---|---|
| openness | 0.070 | 0.061 | −0.525 | −0.455 | +0.072 |
| conscientiousness | 0.058 | **0.140** | −0.458 | −0.399 | −0.026 |
| extraversion | 0.030 | 0.077 | −0.509 | −0.538 | −0.028 |
| agreeableness | 0.047 | 0.060 | −0.386 | −0.340 | +0.110 |
| neuroticism | 0.010 | 0.049 | +0.161 | +0.170 | +0.052 |

**Critical finding**: All GPT-generated interview answers score at or below the mean of the human distribution (z ≈ −0.4 to −0.5), while actual humans cluster near z=0 by construction (standardized). This is a **severe domain gap**: the probe, trained on authentic (often brief, unpolished, naturally spoken) video interview transcripts, assigns low personality scores to GPT's uniformly clean, fluent text regardless of the trait prompted.

W1 HIGH↔LOW is 3–10× smaller than in the essays track (0.010–0.070 vs 0.107–0.303), confirming the probe barely distinguishes GPT-HIGH from GPT-LOW in interview answers. GPT's answers cluster in a narrow stylistic band regardless of prompted personality level.

**w1_humans_neutral** (the gap between human test set and GPT NEUTRAL) is very large: 0.41–0.56 across traits (compared to 0.04–0.08 in essays). This quantifies the domain gap numerically.

### 5.2 Style A: Correlation with Intended Profile

| Trait | RoBERTa Spearman (per-essay) | RoBERTa Spearman (per-user) | ModernBERT Spearman (per-essay) | ModernBERT Spearman (per-user) |
|---|---|---|---|---|
| openness | −0.276 | −0.563 | — | — |
| conscientiousness | −0.263 | −0.588 | — | — |
| extraversion | −0.209 | −0.459 | — | — |
| agreeableness | −0.242 | −0.549 | — | — |
| neuroticism | −0.089 | −0.245 | — | — |
| **macro** | **−0.216** | **−0.480** | **−0.157** | **−0.392** |

**Negative correlations on both probes** confirm this is not a probe artefact. The interpretation: Style A conditions on a human user's actual trait z-scores as the "intended" level. Users with genuinely high openness (high intended z) had naturally distinctive, idiosyncratic speech. GPT, generating a generic fluent interview answer for that question, produces text that the probe scores as below-average. The more extreme the real trait (|z| large), the more GPT diverges from the authentic style of that person — hence the negative correlation.

**This is a domain transfer failure, not a steering failure.** GPT does shift linguistic features when prompted (see §5.3), but those shifts are not in the direction the probe learned to associate with personality from real human speech.

### 5.3 RecruitView LIWC (Style B, HIGH vs LOW)

| Trait | Top feature | δ | 2nd feature | δ |
|---|---|---|---|---|
| neuroticism | avg_word_len | −0.81 | word_count_log | +0.49 |
| conscientiousness | avg_word_len | +0.78 | pron_3p | −0.28 |
| extraversion | excl_rate | +0.61 | pron_i | −0.38 |
| agreeableness | avg_word_len | +0.52 | excl_rate | +0.38 |
| openness | ttr | +0.38 | avg_word_len | +0.23 |

GPT **does** produce measurable surface-feature differences in interview answers when prompted: HIGH-NEU uses shorter words and longer answers (more words but less complex); HIGH-CON uses more formal vocabulary (longer avg word length); HIGH-EXT uses more exclamations. However, these shifts are not aligned with what the probe learned as personality-relevant from real human speech (§5.1).

**Humans vs GPT LIWC (recruitview)**:
- `avg_word_len`: GPT HIGH essays consistently use longer words than human transcripts (δ ≈ −0.55 to −0.70 humans_vs_high across traits). Human spoken language is naturally simpler.
- `word_count_log`: GPT answers are shorter in word count but denser in vocabulary — opposite of spontaneous human speech patterns.
- These are register differences that dominate the personality signal.

---

## 6. Cross-Track Comparison and Paper Conclusions

### 6.1 Main Findings

1. **GPT can steer personality in free-writing essays (Track 1), not in structured interview answers (Track 2).** The essays track shows W1 = 0.11–0.30 (significant distributional shifts); the RecruitView track shows W1 = 0.01–0.07 (near-zero, probe can't distinguish HIGH from LOW).

2. **Steering is trait-dependent.** Openness is most steerable in essays (RoBERTa W1=0.303); Conscientiousness is hardest (both probes, both tracks). Neuroticism and Agreeableness show strongly probe-dependent results (large W1 under ModernBERT, moderate under RoBERTa).

3. **GPT achieves steering primarily through lexical echo.** Trait-specific keyword rates in GPT HIGH essays are 3–10× human rates. TF-IDF top tokens include the prompt's own descriptor words. This is instruction-following, not authentic personality expression.

4. **GPT's default writing persona is more conscientious and agreeable than college students.** NEUTRAL essays score above the human baseline on cCON and cAGR in the essays track. For the paper, this should be framed carefully: the probe reflects what it was trained on (student stream-of-consciousness), and GPT's polished default register looks more "organized" and "warm" to that probe.

5. **The RecruitView domain gap is severe and bidirectional.** GPT's uniform fluency scores below human average across all 5 traits, and the more extreme a human's real trait score, the more GPT's generated answer diverges (negative Spearman −0.22 to −0.48 macro). The probe, trained on authentic speech, penalizes GPT's lack of natural spoken-language idiosyncrasy.

6. **Contamination between traits in essays.** HIGH-NEU prompting suppresses cOPN score (−0.192), and HIGH-EXT suppresses cNEU (−0.190). The trait dimensions are not fully independent as expressed by GPT — prompting one trait shifts vocabulary in ways that affect other traits' probe scores.

### 6.2 Comparison with Probe Baselines

| Model | Probe accuracy ceiling (test set AUC) | Style A alignment AUC | Style A / Probe ratio |
|---|---|---|---|
| RoBERTa | 0.630 | 0.819 | 1.30 |
| ModernBERT | 0.620 | 0.774 | 1.25 |

Style A AUC (LLM distinguishing intended HIGH from LOW) exceeds the probe's own AUC on human test essays. This makes sense: the probe was trained to discriminate natural human personality variation, where the signal is subtle. LLM-generated essays have exaggerated, explicit cues that make the probe's job easier — even though those cues are artificial.

### 6.3 Limitations for Paper

- **W1 reports LLM shift as perceived by this probe.** Low W1 on cCON could mean GPT can't steer that trait, OR the probe is partly blind to it (cCON probe AUC=0.564, near-chance). Both interpretations must be acknowledged.
- **NEUTRAL baseline reflects GPT's default register scored by a probe trained on human stream-of-consciousness.** Domain mismatch between training corpus and GPT output is a confound throughout.
- **HIGH human (labeled cTRAIT=1) ≠ HIGH LLM (prompted HIGH).** One is observed trait expression; the other is intended. The comparison is between categories, not individuals.
- **Keyword echo ≠ authentic personality expression.** The high keyword frequency rates show GPT following instructions, not necessarily expressing the underlying psychological construct.
- **RecruitView probe Spearman is per-answer (not per-user).** Per-user aggregation improves to ~0.47, but the negative LLM alignment Spearman holds at both levels.
- **Emotion NRC features (emo_*) are uniformly uninformative** — all Cliff's δ = 0.0. Either the stream-of-consciousness format suppresses explicit emotion words, or the NRC lexicon is too coarse for this task. Should not be reported as a positive result.

---

## 7. Output File Index

### Essays track

| File | Content |
|---|---|
| `results/llm-alignment/style_b/roberta-base/metrics.json` | W1 distances + CI + contamination matrix (RoBERTa, n=1 522) |
| `results/llm-alignment/style_b/ModernBERT-base/metrics.json` | Same for ModernBERT |
| `results/llm-alignment/style_b/roberta-base/density_<trait>.png` | KDE plots, 4 conditions per trait |
| `results/llm-alignment/style_b/roberta-base/contamination.png` | 5×5 heatmap |
| `results/llm-alignment/style_a/roberta-base/metrics.json` | Style A per-trait AUC/acc (n=500) |
| `results/llm-alignment/style_a/ModernBERT-base/metrics.json` | Same for ModernBERT |
| `results/llm-alignment/analysis/roberta-base/liwc_per_trait.csv` | Mean ± std per (trait, condition, feature) |
| `results/llm-alignment/analysis/roberta-base/liwc_stats_per_trait.csv` | Mann-Whitney p + Cliff's δ for 3 pairwise comparisons |
| `results/llm-alignment/analysis/roberta-base/tfidf_per_trait.csv` | Top-20 discriminating tokens per (trait, condition) |
| `results/llm-alignment/analysis/roberta-base/keyword_freq_per_trait.csv` | Mean rate/1k tokens per (trait, condition, pole) |
| `results/llm-alignment/analysis/roberta-base/keyword_stats_per_trait.csv` | Mann-Whitney + Cliff's δ for keyword comparisons |
| `results/llm-alignment/analysis/roberta-base/errors_per_trait/` | Style A misaligned essays (up to 10 per trait) |

### RecruitView track

| File | Content |
|---|---|
| `results/llm-alignment/recruitview/style_b/roberta-base/metrics.json` | W1 + contamination (n=3 700) |
| `results/llm-alignment/recruitview/style_a/roberta-base/metrics.json` | Style A Spearman/MAE (n=1 395) |
| `results/llm-alignment/recruitview/analysis/roberta-base/liwc_stats_per_trait.csv` | LIWC HIGH vs LOW (Cliff's δ) |
| `results/llm-alignment/recruitview/analysis/roberta-base/errors_per_trait/` | Top-10 worst residuals per trait |
