# Evaluating Personality Expression in Large Language Models

TAR 2025/26 course project (UNIZG FER) — team **TARstars**.

We train a multi-label Big Five classifier on the Pennebaker & King 1999 *Essays* dataset, then use GPT-4o-mini to generate stream-of-consciousness essays conditioned on explicit personality profiles. We compare predicted vs. intended profiles to quantify how faithfully an LLM expresses personality in language.

See [`docs/plan.md`](docs/plan.md) (or the plan checked into `.claude/plans/`) for the full implementation plan.

## Repo layout

```
code/
  datasets/
    essays.csv              # Pennebaker & King 1999 (2467 rows, 5 binary Big Five labels)
    splits/                 # train/val/test index files (generated)
    llm_generated/          # JSONL of GPT-4o-mini essays per prompt style (generated)
  src/                      # Python package — all implementation lives here
  notebooks/                # EDA + final results
  paper/                    # LaTeX source for the submission
requirements.txt
.env.example                # template for OpenAI key
```

## Setup

The project supports Windows, macOS, and Linux. Python 3.10+ is required (3.11 recommended).

### 1. Clone and enter the repo

```bash
git clone <repo-url>
cd evaluating-personality-expression-in-llms
```

### 2. Create and activate a virtual environment

**macOS / Linux (bash / zsh):**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
If activation is blocked, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

**Windows (cmd):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt    # optional, only needed for the local notebooks
```

For NVIDIA GPU on Linux / Windows, install the matching CUDA wheel of PyTorch **before** the line above, e.g.:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```
On Apple Silicon, the default wheel uses MPS automatically.

**On Google Colab**, install **only** `requirements.txt` — Colab provides its own Jupyter stack and the dev requirements conflict with Colab's pinned `kernel-gateway`.

### 4. Configure the OpenAI key

```bash
cp .env.example .env          # macOS / Linux
copy .env.example .env        # Windows cmd
Copy-Item .env.example .env   # Windows PowerShell
```
Then edit `.env` and paste the key from the team's OpenAI project (under the TakeLab organization).

### 5. (one-off) Download NLTK data used by the LIWC-style baseline

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## Running the pipeline

All scripts are entry points on the `src` package. Run them from the `code/` directory:

```bash
cd code

# Data prep — produces datasets/splits/{train,val,test}.txt
python -m src.data

# Baselines (dummy, TF-IDF, LIWC-style)
python -m src.baselines

# Train DeBERTa-v3-base classifier
python -m src.classifier --train

# Generate LLM essays (dry-run first!)
python -m src.generate --style trait-list --limit 20    # dry-run
python -m src.generate --style trait-list               # full run

# Evaluate alignment
python -m src.evaluate --style trait-list

# RQ3 analysis (SHAP + LIWC stats)
python -m src.analyze
```

Each script is restartable and writes outputs atomically. Generated artifacts go to `code/datasets/{splits,llm_generated,results,checkpoints}/`.

## Reproducibility

- Fixed seed `42` for splits and training (see `code/src/config.py`).
- Three seeds reported for the classifier; mean ± std in the paper.
- All randomness in the LLM generation is deferred to OpenAI; we log `model`, `temperature`, and full `raw_response` for every call.

## Team

- Nikola Bačić — 0036551191
- Vedran Kumanović — 1191247427
- Adam Vuković — 0035235027
