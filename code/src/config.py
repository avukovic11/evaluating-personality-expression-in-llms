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

# Track 2 (RECRUITVIEW)
RECRUITVIEW_SPLITS_DIR = DATA_DIR / "splits_recruitview"
LLM_OUTPUTS_RV_DIR = DATA_DIR / "llm_generated_recruitview"


def ensure_dirs() -> None:
    """Create all output directories if they do not yet exist. Idempotent."""
    for d in (
        SPLITS_DIR, LLM_OUTPUTS_DIR, CHECKPOINTS_DIR, RESULTS_DIR,
        RECRUITVIEW_SPLITS_DIR, LLM_OUTPUTS_RV_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


# --- Label conventions --------------------------------------------------------
# Track 1 — Pennebaker Essays. Column order in essays.csv: cEXT, cNEU, cAGR, cCON, cOPN.
TRAIT_COLS = ["cEXT", "cNEU", "cAGR", "cCON", "cOPN"]
TRAIT_NAMES = {
    "cEXT": "Extraversion",
    "cNEU": "Neuroticism",
    "cAGR": "Agreeableness",
    "cCON": "Conscientiousness",
    "cOPN": "Openness",
}

# Track 2 — RECRUITVIEW. HuggingFace dataset columns are lowercase, OCEAN order.
RECRUITVIEW_TRAIT_COLS = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
]
RECRUITVIEW_TRAIT_NAMES = {
    "openness": "Openness",
    "conscientiousness": "Conscientiousness",
    "extraversion": "Extraversion",
    "agreeableness": "Agreeableness",
    "neuroticism": "Neuroticism",
}

# --- Modeling -----------------------------------------------------------------
# RoBERTa-base instead of DeBERTa-v3-base: transformers v5 has a regression in
# DeBERTa's LayerNorm beta/gamma -> weight/bias rename that causes NaN gradients
# within ~50 fine-tuning steps. RoBERTa is well-behaved, slightly smaller (~125M
# vs 184M params), and is the standard encoder baseline in this literature.
CLASSIFIER_MODEL = "roberta-base"
MAX_SEQ_LEN = 512
SEED = 42

# --- LLM generation -----------------------------------------------------------
GEN_MODEL = "gpt-4o-mini"          # primary generator (cheap, ~$2 for full 2467)
GEN_MODEL_SANITY = "gpt-4o"        # scale-check on a ~200-essay subset
GEN_TEMPERATURE = 0.9
PROMPT_STYLES = ("trait-list", "persona", "few-shot")
