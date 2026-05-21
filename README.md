# EuroMillions Crowd Analyser

A full-stack tool that generates, analyses, and tracks EuroMillions number picks — combining human cognitive bias modelling, statistical ensemble prediction, and trained ML models.

> **Disclaimer:** EuroMillions draws are certified truly random using hardware RNG. No model can improve your odds of winning. This tool analyses *human selection behaviour* to help you choose less popular numbers, reducing the chance of splitting a jackpot.

---

## Quick Start

```bash
# 1. Create Python venv and install dependencies
./setup.sh

# 2. Download full draw history (~1200 draws, 2004–2024)
.venv/bin/python scripts/fetch_draws.py --full

# 3. Start the Python backend (port 8000)
.venv/bin/uvicorn engine.server:app --reload

# 4. In a separate terminal, start the Next.js frontend (port 3000)
cd frontend && npm install && npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

---

## Architecture

```
lottery_hack/
├── engine/                  # Python FastAPI backend
│   ├── server.py            # API routes
│   ├── bias.py              # Cognitive bias popularity model
│   ├── generator.py         # Inverse-popularity weighted pick generator
│   ├── predictor.py         # 10-model statistical ensemble
│   ├── models.py            # External ML model runner (LSTM, RF, XGBoost, NN, Linear)
│   └── data.py              # JSON persistence + CSV import + stats
│
├── frontend/                # Next.js 14 (App Router) + TypeScript
│   ├── app/                 # Page and API proxy routes
│   ├── components/tabs/     # One component per UI tab
│   └── lib/                 # bias.ts (mirrors bias.py), types, API client
│
├── scripts/
│   ├── fetch_draws.py       # Downloads FDJ + UK draw history
│   ├── predict_and_notify.py# Generates predictions + posts to Slack
│   └── accuracy_agent.py   # Compares predictions to draws via Claude API
│
├── data/                    # JSON data store (git-ignored)
│   ├── draw_history.json
│   ├── my_picks.json
│   └── predictions.json
│
└── .github/workflows/
    └── predictions.yml      # Mon + Thu cron: auto-predict + commit
```

**Data flow:** Next.js API routes proxy to FastAPI → engine modules → `data/` JSON files.

---

## Features

### Generate tab
Generates picks weighted *inversely* by historical human selection popularity. Popular numbers (7, 17, low numbers, birthday range) are penalised; rare numbers are favoured. Each pick gets a **Uniqueness Score** (0–100): the higher, the fewer humans likely chose the same combination.

### Analyze tab
Enter any 5 numbers + 2 stars and get a full popularity breakdown — which cognitive bias factors (birthday range, lucky 7 family, round numbers, unlucky 13 avoidance) affect each number and how much.

### History tab
Browse your saved picks. Each pick is matched against subsequent draws to show the best historical match (how many numbers/stars came up in a later draw).

### Predict tab
Runs a 10-model statistical ensemble across your draw history:

| Model | What it measures |
|---|---|
| `frequency_regression` | Numbers below expected frequency in last 50 draws |
| `overdue` | Gap since last appearance (up to 20-draw cap) |
| `hot_streak` | Numbers above expected frequency in last 20 draws |
| `recency_decay` | Exponential decay weighting (λ=0.92) |
| `pair_frequency` | Co-occurrence strength in last 100 draws |
| `odd_even_balance` | Match historical odd/even ratio |
| `decade_balance` | Spread across decades 1–10, 11–20, … 41–50 |
| `consecutive_gap` | Historically common consecutive number pairs |
| `sum_proximity` | Numbers that push the pick sum toward historical mean |
| `star_overdue` | Star gap since last appearance |

Each prediction includes per-number signal explanations and a disclaimer.

### Models tab
Runs the trained ML models from `../jupyter-scripts/euromillions/`:

| Model | File | Notes |
|---|---|---|
| LSTM | `lstm_lottery_model_new_11.keras` | Requires TensorFlow |
| Random Forest | `rf_model2.pkl` | 200 trees |
| XGBoost | `xgboost_model2.pkl` + `scaler.pkl` | 5 per-position regressors |
| Neural Network | `nn_model.pkl` | Scikit-learn MLP |
| Linear | `linear_model.pkl` | Regression baseline |

Run models individually or in **Ensemble mode** — all selected models vote on each number and produce a consensus heatmap showing how many models favour each number.

Monte Carlo sampling with Gaussian noise generates diverse picks from deterministic models. Stars are sampled inversely by historical star popularity (none of the ML models predict stars).

### Stats tab
Draw history statistics: frequency distribution of all 50 numbers and 12 stars, historical sum distribution, odd/even ratios, decade spread.

### Why tab
Explains the cognitive bias model and uniqueness score calculation.

---

## Data Sources

Draw history is downloaded from two sources in `scripts/fetch_draws.py`:

1. **FDJ (Française des Jeux)** — ZIP archives covering 2004–2024 (~1200 draws). Four known working archive codes are hardcoded (`200402`, `201402`, `201902`, `202002`).
2. **UK National Lottery XML** — latest single draw, used for incremental updates.

The script is smart about caching: it skips re-downloading if local data already covers the last expected draw (EuroMillions draws on Tuesday and Friday).

```bash
# Download full history (first run)
python scripts/fetch_draws.py --full

# Incremental update (subsequent runs)
python scripts/fetch_draws.py

# Force re-download
python scripts/fetch_draws.py --force

# Check without downloading
python scripts/fetch_draws.py --dry-run
```

---

## GitHub Actions — Auto Predictions

`.github/workflows/predictions.yml` runs every **Monday** and **Thursday** at 12:00 UTC, generating predictions before the Tuesday and Friday draws.

**What it does:**
1. Checks out the repo
2. Installs Python dependencies
3. Fetches the latest draw history
4. Runs `predict_and_notify.py` (all predictors + available ML models → Slack)
5. Commits updated `data/predictions.json` and `data/draw_history.json` back to the repo

**Setup:**

1. Push the repo to GitHub
2. Go to **Settings → Secrets and variables → Actions** and add:
   - `SLACK_WEBHOOK_URL` — your Slack incoming webhook URL
   - `ANTHROPIC_API_KEY` — your Anthropic API key (used by `accuracy_agent.py`)
3. Trigger manually via **Actions → Generate Predictions → Run workflow** to test

---

## Accuracy Agent

`scripts/accuracy_agent.py` compares every saved prediction against subsequent draws and sends the results to **Claude claude-sonnet-4-6** for analysis and improvement suggestions.

```bash
ANTHROPIC_API_KEY=sk-ant-... python scripts/accuracy_agent.py
```

Output is saved to `data/analysis_YYYYMMDD.md`.

The agent computes:
- Per-source average number and star matches
- Overall match distribution (`0n+0s`, `1n+0s`, …)
- Best match per pick

Claude then suggests improvements to ensemble weights, model selection, and pick diversity.

---

## Environment Variables

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

| Variable | Purpose |
|---|---|
| `SLACK_WEBHOOK_URL` | Incoming webhook for Slack notifications |
| `ANTHROPIC_API_KEY` | Claude API key for the accuracy agent |

`.env` is git-ignored. Never commit real secrets.

---

## ML Model Setup

The ML models live outside this repo at `../jupyter-scripts/euromillions/`. If you have them:

```
../jupyter-scripts/euromillions/
├── lstm_lottery_model_new_11.keras
├── lstm_scalers.pkl
├── rf_model2.pkl
├── xgboost_model2.pkl
├── scaler.pkl
├── nn_model.pkl
└── linear_model.pkl
```

The Models tab will show each model's availability. LSTM additionally requires TensorFlow:

```bash
.venv/bin/pip install tensorflow
```

---

## Cognitive Bias Model

The bias model (`engine/bias.py` / `frontend/lib/bias.ts`) assigns a **popularity score** (0–100) to every number and star based on documented human selection patterns:

**Number biases:**
- Birthday range (1–31): +35
- Month range (1–12): +15
- Low number gravity (1–10): +18
- Lucky 7 family (7, 17, 27, 37, 47): +20
- Lucky 7 apex (7): +15
- Lucky 3 family (3, 13, 23, 33, 43): +8
- Round numbers (10, 20, 30, 40, 50): +12
- Fibonacci numbers: +12
- High range (40–50): −12
- Very high range (45–50): −8
- Unlucky 13 avoidance: −25

**Star biases:**
- Low stars (1–6): +30
- Very low stars (1–3): +20
- Lucky star 7: +15
- High stars (9–12): −20
- Very high stars (11–12): −12

**Uniqueness score** = `round(100 - mean(popularities of chosen numbers + stars))`

The same model is implemented identically in Python and TypeScript so the frontend can compute scores without a round-trip to the backend.
