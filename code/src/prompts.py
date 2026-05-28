"""Prompt templates for the LLM generation phase.

Two styles, each instantiated per dataset:
  - **Style D** (single-trait isolated, primary): for each trait T and each level
    {HIGH, LOW, NEUTRAL}, build a prompt that conditions on T alone. NEUTRAL
    omits trait language entirely; HIGH/LOW name the trait pole and follow with
    a short descriptor list. Within each trait the HIGH/LOW descriptors are kept
    similar in length to avoid prompt-length confounds.
  - **Style A** (full multi-trait paired, secondary): list all 5 traits at the
    human profile levels in a single prompt. Tests realistic multi-trait control.
    For RECRUITVIEW the profile is discretized at ±0.5σ → HIGH/MID/LOW.

Each dataset has its own system message and length target:
  - Pennebaker Essays  → 20-minute stream-of-consciousness, 650–700 words.
  - RECRUITVIEW        → spoken interview answer, 60–100 words.

The trait descriptors themselves (TRAIT_POLES) are personality properties and
shared across datasets, keyed by full lowercase trait name.

Style D prompts come in two variants:
  - "full"        → "as someone who is highly extraverted — outgoing,
                     energetic, and socially engaged" (default)
  - "label-only"  → "as someone who is highly extraverted"
Comparing the two disentangles lexical instruction-following (parroting the
descriptor words) from genuine personality-style steering.

Dump example prompts:
    python -m src.prompts
"""

from __future__ import annotations

from typing import Literal

from . import config

# ---------------------------------------------------------------------------
# System messages — one per dataset register
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a college student doing a 20-minute free-writing exercise. "
    "Write the essay in first person — no commentary, no analysis of "
    "personality, no headings, no bullet lists. Aim for 650–700 words "
    "of natural stream-of-consciousness writing."
)

SYSTEM_PROMPT_RECRUITVIEW = (
    "You are a college student doing a mock job interview as part of a "
    "research study. Speak naturally in first person, as if recording a "
    "short clip — no preamble, no "
    "headings or lists. Aim for 60–100 words."
)

# ---------------------------------------------------------------------------
# Per-trait labels and descriptors for Style D HIGH/LOW
# ---------------------------------------------------------------------------

# Single source of truth, keyed by full lowercase trait name. Used by both
# the essays-style and recruitview-style prompt builders.
_POLES_BY_NAME: dict[str, dict[str, str]] = {
    "extraversion": {
        "high_label": "highly extraverted",
        "high_descriptor": "outgoing and energized by people, talkative, and socially bold",
        "low_label": "very introverted",
        "low_descriptor": "quiet and inward-focused, reserved in groups, and drained by socializing",
    },
    "neuroticism": {
        "high_label": "highly neurotic",
        "high_descriptor": "anxious and emotionally reactive, prone to worry and rumination",
        "low_label": "very emotionally stable",
        "low_descriptor": "calm and resilient, rarely upset, and steady under pressure",
    },
    "agreeableness": {
        "high_label": "very agreeable",
        "high_descriptor": "warm and trusting, cooperative, and considerate of others' feelings",
        "low_label": "rather disagreeable",
        "low_descriptor": "blunt and skeptical, competitive, and quick to challenge others",
    },
    "conscientiousness": {
        "high_label": "highly conscientious",
        "high_descriptor": "organized and disciplined, dutiful, and careful about plans and deadlines",
        "low_label": "very disorganized",
        "low_descriptor": "impulsive and unstructured, easily distracted, and careless with obligations",
    },
    "openness": {
        "high_label": "highly open to experience",
        "high_descriptor": "intellectually curious and imaginative, drawn to ideas and new experiences",
        "low_label": "very conventional",
        "low_descriptor": "practical and tradition-minded, preferring routine over abstract speculation",
    },
}

# Essays-style view: keyed by cEXT/cNEU/cAGR/cCON/cOPN.
TRAIT_POLES: dict[str, dict[str, str]] = {
    t: _POLES_BY_NAME[config.TRAIT_NAMES[t].lower()] for t in config.TRAIT_COLS
}

# RECRUITVIEW view: keyed by full lowercase name (matches dataset columns).
RECRUITVIEW_TRAIT_POLES: dict[str, dict[str, str]] = {
    t: _POLES_BY_NAME[t] for t in config.RECRUITVIEW_TRAIT_COLS
}

# Lower-case trait names for natural prompt phrasing.
_TRAIT_NAME_LOWER = {k: v.lower() for k, v in config.TRAIT_NAMES.items()}

Level = Literal["HIGH", "LOW", "NEUTRAL"]
LEVELS: tuple[Level, ...] = ("HIGH", "LOW", "NEUTRAL")

# Style A on RECRUITVIEW discretizes continuous z-scores at ±0.5σ → 3 bins.
LevelA = Literal["HIGH", "MID", "LOW"]
LEVELS_A_RECRUITVIEW: tuple[LevelA, ...] = ("HIGH", "MID", "LOW")

# Style D descriptor variants:
#   "full"        — "as someone who is highly extraverted — outgoing, energetic,
#                    and socially engaged." Lexically primes the LLM with words
#                    the probe was trained to detect.
#   "label-only"  — "as someone who is highly extraverted." Tests whether the
#                    trait label alone shifts style, without LIWC-style priming.
# Run both and compare W1(HIGH, LOW) to disentangle lexical instruction-following
# from genuine style steering.
PromptVariant = Literal["full", "label-only"]
PROMPT_VARIANTS: tuple[PromptVariant, ...] = ("full", "label-only")


# ---------------------------------------------------------------------------
# Style D — single-trait isolated
# ---------------------------------------------------------------------------

def style_d_user_prompt(
    trait: str, level: Level, variant: PromptVariant = "full",
) -> str:
    """Build the Style D user prompt for (trait, level).

    NEUTRAL omits trait language entirely so that all 5 traits' NEUTRAL prompts
    are identical — they give a single baseline distribution for GPT-4o-mini's
    default personality.

    `variant="full"` includes the LIWC-style descriptor after the trait label;
    `variant="label-only"` drops it to test trait-label-only steering.
    """
    if level == "NEUTRAL":
        return (
            "Write a 650–700 word stream-of-consciousness essay about whatever "
            "comes to mind."
        )
    if trait not in TRAIT_POLES:
        raise ValueError(
            f"Unknown trait {trait!r}; expected one of {list(TRAIT_POLES)}."
        )
    pole = TRAIT_POLES[trait]
    label = pole["high_label" if level == "HIGH" else "low_label"]
    descriptor = pole["high_descriptor" if level == "HIGH" else "low_descriptor"]
    suffix = f" — {descriptor}" if variant == "full" else ""
    return (
        f"Write a 650–700 word stream-of-consciousness essay about whatever "
        f"comes to mind, as someone who is {label}{suffix}."
    )


# ---------------------------------------------------------------------------
# Style A — full multi-trait paired
# ---------------------------------------------------------------------------

def style_a_user_prompt(profile: dict[str, int]) -> str:
    """Build the Style A user prompt from a 5-bit binary profile (y=1, n=0).

    Profile must contain all `config.TRAIT_COLS` keys with values in {0, 1}.
    Each trait is rendered as 'HIGH/LOW on <trait_name>'.
    """
    missing = set(config.TRAIT_COLS) - profile.keys()
    if missing:
        raise ValueError(f"Profile missing traits: {missing}")
    parts: list[str] = []
    for t in config.TRAIT_COLS:
        v = profile[t]
        if v not in (0, 1):
            raise ValueError(f"Trait {t} must be 0 or 1; got {v!r}.")
        level = "HIGH" if v == 1 else "LOW"
        parts.append(f"{level} on {_TRAIT_NAME_LOWER[t]}")
    trait_str = ", ".join(parts[:-1]) + ", and " + parts[-1]
    return (
        f"Write a 650–700 word stream-of-consciousness essay as someone who "
        f"scores {trait_str}. Just write whatever comes to mind for 20 minutes "
        f"— informal, no structure required."
    )


# ---------------------------------------------------------------------------
# RECRUITVIEW — Style D (single-trait isolated, interview-answer register)
# ---------------------------------------------------------------------------

def style_d_recruitview_prompt(
    trait: str, level: Level, question: str,
    variant: PromptVariant = "full",
) -> str:
    """Build the Style D user prompt for a RECRUITVIEW interview answer.

    The interview question is included verbatim so the model has something
    concrete to answer; the trait conditioning is layered on top. NEUTRAL
    omits trait language so the 5 traits collapse to one baseline pool.

    `variant="full"` includes the LIWC-style descriptor after the trait label;
    `variant="label-only"` drops it to test trait-label-only steering.
    """
    question = question.strip().rstrip("?") + "?"
    if level == "NEUTRAL":
        return (
            f"Interview question: {question}\n\n"
            f"Answer this question in 60–100 words."
        )
    if trait not in RECRUITVIEW_TRAIT_POLES:
        raise ValueError(
            f"Unknown trait {trait!r}; expected one of "
            f"{list(RECRUITVIEW_TRAIT_POLES)}."
        )
    pole = RECRUITVIEW_TRAIT_POLES[trait]
    label = pole["high_label" if level == "HIGH" else "low_label"]
    descriptor = pole["high_descriptor" if level == "HIGH" else "low_descriptor"]
    suffix = f" — {descriptor}" if variant == "full" else ""
    return (
        f"Interview question: {question}\n\n"
        f"Answer this question in 60–100 words as someone who is "
        f"{label}{suffix}."
    )


# ---------------------------------------------------------------------------
# RECRUITVIEW — Style A (full multi-trait paired, HIGH/MID/LOW)
# ---------------------------------------------------------------------------

# Natural-language fragments for each (trait, discretized-level) combination
# in the Style A multi-trait prompt. Two variant tables, parallel to the
# Style D `variant` parameter: "full" includes the LIWC-style descriptor in
# parens; "label-only" drops it.
_A_RV_FRAGMENTS_FULL: dict[str, dict[str, str]] = {}
_A_RV_FRAGMENTS_LABEL_ONLY: dict[str, dict[str, str]] = {}
for _t in config.RECRUITVIEW_TRAIT_COLS:
    _pole = RECRUITVIEW_TRAIT_POLES[_t]
    _A_RV_FRAGMENTS_FULL[_t] = {
        "HIGH": f"high on {_t} ({_pole['high_descriptor']})",
        "LOW":  f"low on {_t} ({_pole['low_descriptor']})",
        "MID":  f"around average on {_t}",
    }
    _A_RV_FRAGMENTS_LABEL_ONLY[_t] = {
        "HIGH": f"high on {_t}",
        "LOW":  f"low on {_t}",
        "MID":  f"around average on {_t}",
    }
del _t, _pole


def discretize_z(z: float, threshold: float = 0.5) -> LevelA:
    """Map a z-score to HIGH (z > +thr), LOW (z < -thr), or MID."""
    if z > threshold:
        return "HIGH"
    if z < -threshold:
        return "LOW"
    return "MID"


def style_a_recruitview_prompt(
    levels: dict[str, LevelA], question: str,
    variant: PromptVariant = "full",
) -> str:
    """Build the Style A user prompt from a discretized 5-trait profile.

    `levels` must map every `config.RECRUITVIEW_TRAIT_COLS` key to one of
    "HIGH", "MID", "LOW". `variant` controls descriptor inclusion: "full"
    keeps the LIWC-style descriptor in parens after each trait label,
    "label-only" drops it.
    """
    missing = set(config.RECRUITVIEW_TRAIT_COLS) - levels.keys()
    if missing:
        raise ValueError(f"Profile missing traits: {missing}")
    fragments = (
        _A_RV_FRAGMENTS_FULL if variant == "full"
        else _A_RV_FRAGMENTS_LABEL_ONLY
    )
    parts: list[str] = []
    for t in config.RECRUITVIEW_TRAIT_COLS:
        lvl = levels[t]
        if lvl not in ("HIGH", "MID", "LOW"):
            raise ValueError(
                f"Trait {t} level must be HIGH/MID/LOW; got {lvl!r}."
            )
        parts.append(fragments[t][lvl])
    trait_str = "; ".join(parts[:-1]) + "; and " + parts[-1]
    question = question.strip().rstrip("?") + "?"
    return (
        f"Interview question: {question}\n\n"
        f"Answer this question in 60–100 words as someone who is "
        f"{trait_str}."
    )


# ---------------------------------------------------------------------------
# OpenAI Chat Completions payload helper
# ---------------------------------------------------------------------------

def build_messages(
    user_prompt: str, dataset: str = "essays",
) -> list[dict[str, str]]:
    """Return the `messages=` array for the OpenAI Chat Completions API.

    `dataset` selects the system message: "essays" for the Pennebaker
    free-writing register, "recruitview" for the short interview-answer
    register.
    """
    if dataset == "essays":
        system = SYSTEM_PROMPT
    elif dataset == "recruitview":
        system = SYSTEM_PROMPT_RECRUITVIEW
    else:
        raise ValueError(
            f"Unknown dataset {dataset!r}; expected 'essays' or 'recruitview'."
        )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_prompt},
    ]


# ---------------------------------------------------------------------------
# CLI: dump example prompts for inspection
# ---------------------------------------------------------------------------

_EXAMPLE_RV_QUESTION = (
    "Tell me about a time you had to work with a teammate whose approach "
    "to a project differed sharply from your own. How did you handle it?"
)


def _dump_examples() -> None:
    print("=" * 70)
    print("ESSAYS — SYSTEM MESSAGE")
    print("=" * 70)
    print(SYSTEM_PROMPT)
    print()

    for variant in PROMPT_VARIANTS:
        print("=" * 70)
        print(f"ESSAYS — STYLE D / variant={variant!r} (5 traits × 3 levels)")
        print("=" * 70)
        for trait in config.TRAIT_COLS:
            for level in LEVELS:
                print(f"\n--- {trait} ({config.TRAIT_NAMES[trait]}) / {level} ---")
                print(style_d_user_prompt(trait, level, variant=variant))

    print()
    print("=" * 70)
    print("ESSAYS — STYLE A (example profile)")
    print("=" * 70)
    profile = {"cEXT": 1, "cNEU": 0, "cAGR": 1, "cCON": 1, "cOPN": 0}
    print(f"\nprofile = {profile}")
    print(style_a_user_prompt(profile))

    print()
    print("=" * 70)
    print("RECRUITVIEW — SYSTEM MESSAGE")
    print("=" * 70)
    print(SYSTEM_PROMPT_RECRUITVIEW)
    print()

    for variant in PROMPT_VARIANTS:
        print("=" * 70)
        print(
            f"RECRUITVIEW — STYLE D / variant={variant!r} (5 traits × 3 levels)"
        )
        print("=" * 70)
        print(f"interview question: {_EXAMPLE_RV_QUESTION}")
        for trait in config.RECRUITVIEW_TRAIT_COLS:
            for level in LEVELS:
                print(f"\n--- {trait} / {level} ---")
                print(style_d_recruitview_prompt(
                    trait, level, _EXAMPLE_RV_QUESTION, variant=variant,
                ))

    rv_levels: dict[str, LevelA] = {
        "openness":          "HIGH",
        "conscientiousness": "MID",
        "extraversion":      "HIGH",
        "agreeableness":     "LOW",
        "neuroticism":       "LOW",
    }
    for variant in PROMPT_VARIANTS:
        print()
        print("=" * 70)
        print(f"RECRUITVIEW — STYLE A / variant={variant!r}")
        print("=" * 70)
        print(f"levels = {rv_levels}")
        print(style_a_recruitview_prompt(
            rv_levels, _EXAMPLE_RV_QUESTION, variant=variant,
        ))


if __name__ == "__main__":
    _dump_examples()
