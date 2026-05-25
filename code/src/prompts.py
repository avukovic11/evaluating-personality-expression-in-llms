"""Prompt templates for the LLM generation phase.

Two styles:
  - **Style D** (single-trait isolated, primary): for each trait T and each level
    {HIGH, LOW, NEUTRAL}, build a prompt that conditions on T alone. NEUTRAL
    omits trait language entirely; HIGH/LOW name the trait pole and follow with
    a short descriptor list. Within each trait the HIGH/LOW descriptors are kept
    similar in length to avoid prompt-length confounds.
  - **Style A** (full multi-trait paired, secondary): list all 5 traits at the
    human profile levels in a single prompt. Tests realistic multi-trait control.

Both styles share a system message that steers the model toward the dataset's
stream-of-consciousness register and away from meta-commentary about personality.

Dump example prompts:
    python -m src.prompts
"""

from __future__ import annotations

from typing import Literal

from . import config

# ---------------------------------------------------------------------------
# Shared system message
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a college student doing a 20-minute free-writing exercise. "
    "Write the essay only — no commentary, no analysis of personality, "
    "no headings, no bullet lists. Aim for 600–800 words of natural "
    "stream-of-consciousness writing."
)

# ---------------------------------------------------------------------------
# Per-trait labels and descriptors for Style D HIGH/LOW
# ---------------------------------------------------------------------------

TRAIT_POLES: dict[str, dict[str, str]] = {
    "cEXT": {
        "high_label": "highly extraverted",
        "high_descriptor": "outgoing, energetic, and socially engaged",
        "low_label": "very introverted",
        "low_descriptor": "reserved, prefers solitude, and finds social situations draining",
    },
    "cNEU": {
        "high_label": "highly neurotic",
        "high_descriptor": "anxious, easily stressed, and prone to negative emotions",
        "low_label": "very emotionally stable",
        "low_descriptor": "calm, even-tempered, and resilient under stress",
    },
    "cAGR": {
        "high_label": "very agreeable",
        "high_descriptor": "warm, cooperative, and considerate of others",
        "low_label": "rather disagreeable",
        "low_descriptor": "competitive, critical, and skeptical of others' motives",
    },
    "cCON": {
        "high_label": "highly conscientious",
        "high_descriptor": "organized, disciplined, and dependable",
        "low_label": "very disorganized",
        "low_descriptor": "spontaneous, careless with plans, and easily distracted",
    },
    "cOPN": {
        "high_label": "highly open to experience",
        "high_descriptor": "curious, imaginative, and drawn to novelty",
        "low_label": "very conventional",
        "low_descriptor": "practical, routine-oriented, and prefers the familiar",
    },
}

# Lower-case trait names for natural prompt phrasing.
_TRAIT_NAME_LOWER = {k: v.lower() for k, v in config.TRAIT_NAMES.items()}

Level = Literal["HIGH", "LOW", "NEUTRAL"]
LEVELS: tuple[Level, ...] = ("HIGH", "LOW", "NEUTRAL")


# ---------------------------------------------------------------------------
# Style D — single-trait isolated
# ---------------------------------------------------------------------------

def style_d_user_prompt(trait: str, level: Level) -> str:
    """Build the Style D user prompt for (trait, level).

    NEUTRAL omits trait language entirely so that all 5 traits' NEUTRAL prompts
    are identical — they give a single baseline distribution for GPT-4o-mini's
    default personality.
    """
    if level == "NEUTRAL":
        return (
            "Write a 600–800 word stream-of-consciousness essay about whatever "
            "comes to mind."
        )
    if trait not in TRAIT_POLES:
        raise ValueError(
            f"Unknown trait {trait!r}; expected one of {list(TRAIT_POLES)}."
        )
    pole = TRAIT_POLES[trait]
    label_key, descr_key = (
        ("high_label", "high_descriptor") if level == "HIGH"
        else ("low_label", "low_descriptor")
    )
    return (
        f"Write a 600–800 word stream-of-consciousness essay about whatever "
        f"comes to mind, as someone who is {pole[label_key]} — "
        f"{pole[descr_key]}."
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
        f"Write a 600–800 word stream-of-consciousness essay as someone who "
        f"scores {trait_str}. Just write whatever comes to mind for 20 minutes "
        f"— informal, no structure required."
    )


# ---------------------------------------------------------------------------
# OpenAI Chat Completions payload helper
# ---------------------------------------------------------------------------

def build_messages(user_prompt: str) -> list[dict[str, str]]:
    """Return the `messages=` array for the OpenAI Chat Completions API."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


# ---------------------------------------------------------------------------
# CLI: dump example prompts for inspection
# ---------------------------------------------------------------------------

def _dump_examples() -> None:
    print("=" * 70)
    print("SYSTEM MESSAGE")
    print("=" * 70)
    print(SYSTEM_PROMPT)
    print()

    print("=" * 70)
    print("STYLE D — single-trait isolated (5 traits × 3 levels)")
    print("=" * 70)
    for trait in config.TRAIT_COLS:
        for level in LEVELS:
            print(f"\n--- {trait} ({config.TRAIT_NAMES[trait]}) / {level} ---")
            print(style_d_user_prompt(trait, level))

    print()
    print("=" * 70)
    print("STYLE A — full multi-trait paired (example profile)")
    print("=" * 70)
    profile = {"cEXT": 1, "cNEU": 0, "cAGR": 1, "cCON": 1, "cOPN": 0}
    print(f"\nprofile = {profile}")
    print(style_a_user_prompt(profile))


if __name__ == "__main__":
    _dump_examples()
