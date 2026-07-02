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
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
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


def _claude_prediction(draws: list[dict], signals: dict) -> dict | None:
    """Ask Claude to pick numbers using a contrarian strategy, save as a prediction."""
    if not ANTHROPIC_API_KEY:
        print("  ANTHROPIC_API_KEY not set — skipping Claude prediction")
        return None

    from engine.bias import uniqueness_score
    from engine.data import make_pick

    recent_str = "\n".join(
        f"  {d['date']}: {d['numbers']} ★ {d['stars']}"
        for d in draws[-30:]
    )
    top_nums    = [s["number"] for s in signals["numbers"]]
    overdue     = sorted(signals["numbers"], key=lambda s: s["overdue"], reverse=True)[:8]
    cold        = sorted(signals["numbers"], key=lambda s: s["cold"],    reverse=True)[:8]
    top_stars   = [s["star"] for s in signals["stars"]]

    prompt = f"""You are picking EuroMillions numbers for the next draw.
Goal: choose numbers FEWER people typically pick, to minimise jackpot splits if we win.

Recent 30 draws:
{recent_str}

Statistical signals (from the prediction engine):
- Top ensemble numbers: {top_nums}
- Most overdue:         {[s["number"] for s in overdue]}
- Cold (below frequency): {[s["number"] for s in cold]}
- Overdue stars:        {top_stars}

Human bias to AVOID: birthday range (1-31), lucky 7 family (7,17,27,37,47),
round numbers (10,20,30,40,50), low numbers (1-10).
Prefer: 32-50 range, uncommon gaps, numbers humans instinctively skip.

Pick 5 UNIQUE numbers (1-50) and 2 UNIQUE lucky stars (1-12).

Reply ONLY with this JSON — no markdown, no explanation outside the JSON:
{{"numbers": [n1, n2, n3, n4, n5], "stars": [s1, s2], "reasoning": "one sentence"}}"""

    body = json.dumps({
        "model": "claude-sonnet-4-6",
        "max_tokens": 256,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        text = data["content"][0]["text"].strip()
        # Strip markdown fences if Claude wraps in ```json
        if text.startswith("```"):
            text = "\n".join(
                line for line in text.splitlines()
                if not line.strip().startswith("```")
            ).strip()
        parsed = json.loads(text)
        numbers = sorted(int(n) for n in parsed["numbers"])
        stars   = sorted(int(s) for s in parsed["stars"])
        reasoning = parsed.get("reasoning", "")

        if len(numbers) != 5 or not all(1 <= n <= 50 for n in numbers):
            print(f"  Invalid numbers from Claude: {numbers}")
            return None
        if len(stars) != 2 or not all(1 <= s <= 12 for s in stars):
            print(f"  Invalid stars from Claude: {stars}")
            return None

        pick = make_pick(numbers, stars, uniqueness_score(numbers, stars), source="claude")
        pick["reasoning"] = reasoning
        return pick

    except urllib.error.HTTPError as e:
        print(f"  Claude API error {e.code}: {e.read().decode()[:200]}")
    except json.JSONDecodeError:
        print(f"  Claude returned non-JSON: {text[:120]}")
    except Exception as e:
        print(f"  Claude prediction error: {e}")
    return None


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

    # ── Claude prediction ─────────────────────────────────────────────────────

    print("\n[4/4] Claude prediction…")
    signals = top_signals(draws)
    claude_pick = _claude_prediction(draws, signals)
    if claude_pick:
        all_predictions.append(claude_pick)
        blocks.append({"type": "divider"})
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"*🧠 Claude Prediction*\n"
                    f"• {_pick_line(claude_pick['numbers'], claude_pick['stars'], claude_pick['uniquenessScore'])}\n"
                    f"_{claude_pick.get('reasoning', '')}_"
                ),
            },
        })
        print(f"  Pick: {claude_pick['numbers']} ★ {claude_pick['stars']}")
    else:
        print("  Skipped.")

    # ── Top signals sidebar ───────────────────────────────────────────────────

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
