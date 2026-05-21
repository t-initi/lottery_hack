"""
Standalone prediction + Slack notification script.

Run manually or via GitHub Actions (Mon/Thu cron).
Generates picks from all available predictors and ML models,
appends results to data/predictions.json, and posts a Slack
Block Kit message via SLACK_WEBHOOK_URL.

Requires: SLACK_WEBHOOK_URL in env (or .env file)
Optional: ML model files in ../../jupyter-scripts/euromillions/
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

# ── Bootstrap path so we can import engine modules ────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from engine.data import load_draws, save_draws, load_picks
from engine.predictor import generate_predictions, top_signals, DISCLAIMER
from engine.generator import generate_picks
from engine.models import list_models, run_model

# ── Constants ─────────────────────────────────────────────────────────────────

SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL", "")
DATA_DIR = ROOT / "data"
PREDICTIONS_FILE = DATA_DIR / "predictions.json"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_predictions() -> list[dict]:
    if PREDICTIONS_FILE.exists():
        return json.loads(PREDICTIONS_FILE.read_text())
    return []


def _save_predictions(preds: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    all_preds = _load_predictions()
    existing_ids = {p["id"] for p in all_preds}
    new_preds = [p for p in preds if p["id"] not in existing_ids]
    all_preds.extend(new_preds)
    PREDICTIONS_FILE.write_text(json.dumps(all_preds, indent=2))
    print(f"  Saved {len(new_preds)} new predictions ({len(all_preds)} total)")


def _post_slack(payload: dict) -> None:
    if not SLACK_WEBHOOK_URL:
        print("  SLACK_WEBHOOK_URL not set — skipping Slack notification")
        return
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        SLACK_WEBHOOK_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            print(f"  Slack: {resp.status} {resp.reason}")
    except urllib.error.HTTPError as e:
        print(f"  Slack error {e.code}: {e.read().decode()[:200]}")
    except Exception as e:
        print(f"  Slack error: {e}")


def _ball(n: int) -> str:
    return f"`{n:02d}`"


def _pick_line(numbers: list[int], stars: list[int], score: int) -> str:
    nums = "  ".join(_ball(n) for n in numbers)
    strs = "  ".join(f"★{s}" for s in stars)
    return f"{nums}   {strs}   _(uniqueness: {score})_"


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    now = datetime.now(timezone.utc)
    next_draw = "Tuesday" if now.weekday() < 1 else "Friday"
    print(f"\n=== EuroMillions predictions — {now.strftime('%Y-%m-%d %H:%M UTC')} ===")

    draws = load_draws()
    if len(draws) < 50:
        print("ERROR: Not enough draw history. Run: python scripts/fetch_draws.py --full")
        sys.exit(1)

    all_predictions: list[dict] = []
    blocks: list[dict] = []

    # ── Header ────────────────────────────────────────────────────────────────

    blocks.append({
        "type": "header",
        "text": {
            "type": "plain_text",
            "text": f"🎰 EuroMillions Predictions — {next_draw} draw",
            "emoji": True,
        },
    })
    blocks.append({
        "type": "context",
        "elements": [{"type": "mrkdwn", "text": f"Generated {now.strftime('%a %d %b %Y, %H:%M UTC')} · {len(draws)} draws in history"}],
    })
    blocks.append({"type": "divider"})

    # ── Statistical predictor picks ───────────────────────────────────────────

    print("\n[1/3] Statistical predictor (ensemble)…")
    try:
        stat_preds = generate_predictions(count=3)
        all_predictions.extend(stat_preds)

        lines = [
            f"• {_pick_line(p['numbers'], p['stars'], p['uniquenessScore'])}"
            for p in stat_preds
        ]
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": "*📊 Statistical Ensemble (overdue · recency · pair frequency)*\n" + "\n".join(lines)},
        })
        print(f"  Generated {len(stat_preds)} predictions")
    except Exception as e:
        print(f"  ERROR: {e}")
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"*📊 Statistical Ensemble* — ⚠️ error: {e}"}})

    # ── Random generator picks ────────────────────────────────────────────────

    print("\n[2/3] Bias-weighted generator…")
    try:
        gen_picks = generate_picks(count=2, blend_draws=True)
        all_predictions.extend(gen_picks)

        lines = [
            f"• {_pick_line(p['numbers'], p['stars'], p['uniquenessScore'])}"
            for p in gen_picks
        ]
        blocks.append({"type": "divider"})
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": "*🎲 Bias-Weighted Generator (contrarian · rare numbers)*\n" + "\n".join(lines)},
        })
        print(f"  Generated {len(gen_picks)} picks")
    except Exception as e:
        print(f"  ERROR: {e}")
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"*🎲 Generator* — ⚠️ error: {e}"}})

    # ── ML model picks (available models only) ────────────────────────────────

    print("\n[3/3] ML models…")
    available_models = [m for m in list_models() if m["available"]]
    if available_models:
        blocks.append({"type": "divider"})
        ml_sections: list[str] = []

        for model_meta in available_models:
            mid = model_meta["id"]
            try:
                result = run_model(mid, n_picks=2, n_samples=200, draws=draws)
                all_predictions.extend(result["picks"])
                pick_lines = [
                    f"  • {_pick_line(p['numbers'], p['stars'], p['uniquenessScore'])}"
                    for p in result["picks"]
                ]
                ml_sections.append(f"*{model_meta['label']}*\n" + "\n".join(pick_lines))
                print(f"  {mid}: {len(result['picks'])} picks")
            except Exception as e:
                ml_sections.append(f"*{model_meta['label']}* — ⚠️ {e}")
                print(f"  {mid}: ERROR — {e}")

        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": "*🤖 ML Models*\n" + "\n\n".join(ml_sections)},
        })
    else:
        print("  No ML models available — skipping")

    # ── Top signals sidebar ───────────────────────────────────────────────────

    signals = top_signals(draws)
    top_nums = " ".join(_ball(s["number"]) for s in signals["numbers"][:5])
    top_stars = " ".join(f"★{s['star']}" for s in signals["stars"][:4])

    blocks.append({"type": "divider"})
    blocks.append({
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": f"*📡 Top signals*\nNumbers: {top_nums}\nStars: {top_stars}",
        },
    })

    # ── Disclaimer ────────────────────────────────────────────────────────────

    blocks.append({"type": "divider"})
    blocks.append({
        "type": "context",
        "elements": [{"type": "mrkdwn", "text": f"_{DISCLAIMER}_"}],
    })

    # ── Save + notify ─────────────────────────────────────────────────────────

    _save_predictions(all_predictions)
    print("\nPosting to Slack…")
    _post_slack({"blocks": blocks})
    print("\nDone.")


if __name__ == "__main__":
    main()
