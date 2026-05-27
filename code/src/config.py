"""Project-wide paths, label conventions, and modeling constants.

Always import constants from here instead of hardcoding strings. Paths are
`pathlib.Path` objects so they work identically on Windows, macOS, and Linux.
"""

from pathlib import Path

# --- Filesystem layout ---------------------------------------------------------
# config.py lives at code/src/config.py, so parents[2] is the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = PROJECT_ROOT / "code"
DATA_DIR = CODE_DIR / "datasets"

ESSAYS_CSV = DATA_DIR / "essays.csv"

SPLITS_DIR = DATA_DIR / "splits"
LLM_OUTPUTS_DIR = DATA_DIR / "llm_generated"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints"
RESULTS_DIR = DATA_DIR / "results"


def ensure_dirs() -> None:
    """Create all output directories if they do not yet exist. Idempotent."""
    for d in (SPLITS_DIR, LLM_OUTPUTS_DIR, CHECKPOINTS_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


# --- Label conventions --------------------------------------------------------
# Column order in essays.csv: cEXT, cNEU, cAGR, cCON, cOPN.
TRAIT_COLS = ["cEXT", "cNEU", "cAGR", "cCON", "cOPN"]
TRAIT_NAMES = {
    "cEXT": "Extraversion",
    "cNEU": "Neuroticism",
    "cAGR": "Agreeableness",
    "cCON": "Conscientiousness",
    "cOPN": "Openness",
}

# --- Modeling -----------------------------------------------------------------
# RoBERTa-base instead of DeBERTa-v3-base: transformers v5 has a regression in
# DeBERTa's LayerNorm beta/gamma -> weight/bias rename that causes NaN gradients
# within ~50 fine-tuning steps. RoBERTa is well-behaved, slightly smaller (~125M
# vs 184M params), and is the standard encoder baseline in this literature.
CLASSIFIER_MODEL = "roberta-base"
MAX_SEQ_LEN = 512
SEED = 42

# --- Trait pole vocabulary ----------------------------------------------------
# Single source of truth for the label + descriptor strings used by prompts.py
# to build Style D prompts. Prompt text must NOT change after dry-run essays
# are generated without also regenerating those essays.
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

# --- Trait keyword vocabulary -------------------------------------------------
# Broader word lists for keyword-density analysis in analyze.py. Intentionally
# wider than TRAIT_POLES descriptors — the point is to detect trait vocabulary
# that leaks through beyond what was explicitly prompted.
# Drawn from TRAIT_POLES core terms plus close morphological variants.
TRAIT_KEYWORDS: dict[str, dict[str, list[str]]] = {
    "cEXT": {
        "high": [
            "extraverted", "extravert", "outgoing", "energetic", "energy",
            "social", "sociable", "talkative", "lively", "people", "party",
            "friends", "engaged", "engaging",
        ],
        "low": [
            "introverted", "introvert", "reserved", "solitude", "alone",
            "quiet", "withdrawn", "draining", "drained", "isolated", "shy",
        ],
    },
    "cNEU": {
        "high": [
            "neurotic", "anxious", "anxiety", "stressed", "stress",
            "worry", "worried", "worrying", "nervous", "tense", "negative",
            "overwhelmed", "upset", "fearful", "fear",
        ],
        "low": [
            "stable", "calm", "relaxed", "resilient", "composed",
            "peaceful", "serene", "balanced", "even", "steady",
        ],
    },
    "cAGR": {
        "high": [
            "agreeable", "warm", "cooperative", "considerate", "kind",
            "friendly", "helpful", "compassionate", "caring", "generous",
        ],
        "low": [
            "disagreeable", "competitive", "critical", "skeptical",
            "harsh", "argumentative", "cold", "suspicious", "blunt",
        ],
    },
    "cCON": {
        "high": [
            "conscientious", "organized", "organisation", "organization",
            "disciplined", "discipline", "dependable", "responsible",
            "careful", "thorough", "systematic", "diligent", "planned",
            "schedule", "goal",
        ],
        "low": [
            "disorganized", "spontaneous", "careless", "distracted",
            "impulsive", "messy", "forgetful", "haphazard", "scattered",
        ],
    },
    "cOPN": {
        "high": [
            "open", "curious", "curiosity", "imaginative", "imagination",
            "creative", "creativity", "novelty", "novel", "artistic",
            "intellectual", "adventurous", "inventive",
        ],
        "low": [
            "conventional", "practical", "routine", "familiar",
            "traditional", "ordinary", "conservative", "predictable",
        ],
    },
}

# --- LLM generation -----------------------------------------------------------
GEN_MODEL = "gpt-4o-mini"          # primary generator (cheap, ~$2 for full 2467)
GEN_MODEL_SANITY = "gpt-4o"        # scale-check on a ~200-essay subset
GEN_TEMPERATURE = 0.9
PROMPT_STYLES = ("trait-list", "persona", "few-shot")
