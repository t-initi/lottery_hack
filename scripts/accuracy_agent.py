"""
Claude API accuracy agent.

Loads saved predictions, compares them against subsequent draws,
computes per-model match statistics, then asks Claude for
improvement suggestions. Saves a markdown report to data/.

Usage:
  python scripts/accuracy_agent.py

Requires: ANTHROPIC_API_KEY in env (or .env file)
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from engine.data import load_draws

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
DATA_DIR = ROOT / "data"
PREDICTIONS_FILE = DATA_DIR / "predictions.json"


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_predictions() -> list[dict]:
    if not PREDICTIONS_FILE.exists():
        return []
    return json.loads(PREDICTIONS_FILE.read_text())


# ── Match scoring ─────────────────────────────────────────────────────────────

def _best_match(pick: dict, draws: list[dict]) -> dict | None:
    """Find the draw after this pick's timestamp with the most number matches."""
    pick_date = pick.get("timestamp", "")[:10]
    subsequent = [d for d in draws if d["date"] > pick_date]
    if not subsequent:
        return None

    best = None
    best_score = -1
    for draw in subsequent:
        nm = len(set(pick["numbers"]) & set(draw["numbers"]))
        sm = len(set(pick["stars"]) & set(draw.get("stars", [])))
        score = nm * 2 + sm
        if score > best_score:
            best_score = score
            best = {"draw": draw, "numMatches": nm, "starMatches": sm, "total": score}
    return best


def compute_accuracy(predictions: list[dict], draws: list[dict]) -> dict:
    """
    Returns:
      {
        "per_pick": [{id, source, numMatches, starMatches, draw_date}, ...],
        "per_source": {source: {picks, avg_num, avg_star, max_num}},
        "distribution": {"3n+1s": 5, ...},
        "total_picks": int,
        "draws_available": int,
      }
    """
    per_pick = []
    per_source: dict[str, dict] = defaultdict(lambda: {"picks": 0, "num_sum": 0, "star_sum": 0, "max_num": 0})
    dist: dict[str, int] = defaultdict(int)

    for pred in predictions:
        match = _best_match(pred, draws)
        if match is None:
            continue
        nm = match["numMatches"]
        sm = match["starMatches"]
        src = pred.get("source", "unknown")
        per_pick.append({
            "id": pred["id"],
            "source": src,
            "numbers": pred["numbers"],
            "stars": pred["stars"],
            "numMatches": nm,
            "starMatches": sm,
            "draw_date": match["draw"]["date"],
            "draw_numbers": match["draw"]["numbers"],
        })
        per_source[src]["picks"] += 1
        per_source[src]["num_sum"] += nm
        per_source[src]["star_sum"] += sm
        per_source[src]["max_num"] = max(per_source[src]["max_num"], nm)
        dist[f"{nm}n+{sm}s"] += 1

    for src, s in per_source.items():
        n = max(s["picks"], 1)
        s["avg_num"] = round(s["num_sum"] / n, 2)
        s["avg_star"] = round(s["star_sum"] / n, 2)

    return {
        "per_pick": per_pick,
        "per_source": dict(per_source),
        "distribution": dict(dist),
        "total_picks": len(per_pick),
        "draws_available": len(draws),
    }


# ── Claude API call ───────────────────────────────────────────────────────────

def _call_claude(prompt: str, system: str) -> str:
    if not ANTHROPIC_API_KEY:
        return "(ANTHROPIC_API_KEY not set — AI analysis skipped)"

    payload = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 2048,
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
    }
    body = json.dumps(payload).encode()
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
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
            return data["content"][0]["text"]
    except urllib.error.HTTPError as e:
        return f"(Claude API error {e.code}: {e.read().decode()[:300]})"
    except Exception as e:
        return f"(Claude API error: {e})"


def _build_prompt(accuracy: dict, predictions: list[dict]) -> tuple[str, str]:
    system = (
        "You are a quantitative lottery analyst. "
        "EuroMillions draws are certified truly random, so no prediction strategy can improve "
        "win odds. Your role is to analyse human-facing metrics: whether picks are unique "
        "(to minimise jackpot splits), diverse across strategies, and whether any statistical "
        "signals correlate with slightly better number-match scores (even if purely by chance). "
        "Be concise, honest about randomness, and actionable."
    )

    src_table = "\n".join(
        f"  {src}: avg {s['avg_num']} nums, {s['avg_star']} stars, best {s['max_num']} nums ({s['picks']} picks)"
        for src, s in accuracy["per_source"].items()
    )

    dist_table = "\n".join(
        f"  {k}: {v} picks"
        for k, v in sorted(accuracy["distribution"].items())
    )

    sample_picks = "\n".join(
        f"  [{p['source']}] {p['numbers']} + {p['stars']} → {p['numMatches']}n + {p['starMatches']}s vs {p['draw_numbers']} on {p['draw_date']}"
        for p in accuracy["per_pick"][:10]
    )

    # Include model configs for context
    model_info = ""
    if predictions:
        models_used = set()
        for p in predictions:
            models_used.update(p.get("modelsUsed", []))
        model_info = f"\nModels in statistical ensemble: {', '.join(sorted(models_used))}"

    prompt = f"""Here is accuracy data for EuroMillions prediction picks vs actual draws.

## Summary
- Total evaluated picks: {accuracy['total_picks']}
- Draws in history: {accuracy['draws_available']}

## Per-source performance
{src_table}

## Match distribution (across all picks × subsequent draws)
{dist_table}

## Sample picks vs actuals
{sample_picks}
{model_info}

## Questions for you
1. Which prediction source/strategy shows the strongest (or weakest) correlation with number matches?
2. Are there any patterns in the picks that could improve uniqueness (to reduce jackpot splits)?
3. What changes to the statistical ensemble weights or ML model selection would you suggest?
4. Any other insights from this data?

Please structure your response with clear headings and be specific about what to change in the code."""

    return system, prompt


# ── Report saving ─────────────────────────────────────────────────────────────

def _save_report(accuracy: dict, analysis: str) -> Path:
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = DATA_DIR / f"analysis_{today}.md"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    src_table = "\n".join(
        f"| {src} | {s['picks']} | {s['avg_num']} | {s['avg_star']} | {s['max_num']} |"
        for src, s in accuracy["per_source"].items()
    )
    dist_table = "\n".join(
        f"| {k} | {v} |"
        for k, v in sorted(accuracy["distribution"].items())
    )

    content = f"""# EuroMillions Accuracy Report — {datetime.now(timezone.utc).strftime("%Y-%m-%d")}

## Match Statistics

| Source | Picks | Avg nums | Avg stars | Best nums |
|--------|-------|----------|-----------|-----------|
{src_table}

## Distribution

| Match | Count |
|-------|-------|
{dist_table}

---

## Claude Analysis

{analysis}

---
_Generated by `scripts/accuracy_agent.py` · {datetime.now(timezone.utc).isoformat()}Z_
"""
    path.write_text(content)
    return path


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n=== EuroMillions Accuracy Agent ===\n")

    predictions = _load_predictions()
    draws = load_draws()

    if not predictions:
        print("No predictions found. Run predict_and_notify.py first.")
        sys.exit(0)
    if not draws:
        print("No draw history. Run: python scripts/fetch_draws.py --full")
        sys.exit(1)

    print(f"Loaded {len(predictions)} predictions, {len(draws)} draws")

    print("Computing accuracy…")
    accuracy = compute_accuracy(predictions, draws)
    print(f"  Evaluated {accuracy['total_picks']} picks with subsequent draws")

    if accuracy["total_picks"] == 0:
        print("  No picks have a subsequent draw to compare against yet.")
        print("  Re-run after the next draw.")
        sys.exit(0)

    for src, s in accuracy["per_source"].items():
        print(f"  {src}: avg {s['avg_num']} num matches / {s['picks']} picks")

    print("\nCalling Claude for analysis…")
    system, prompt = _build_prompt(accuracy, predictions)
    analysis = _call_claude(prompt, system)
    print("  Done.")

    report_path = _save_report(accuracy, analysis)
    print(f"\nReport saved to: {report_path}")
    print("\n--- Claude Analysis Preview ---")
    print(analysis[:800])
    if len(analysis) > 800:
        print(f"\n… ({len(analysis)} chars total, see {report_path})")


if __name__ == "__main__":
    main()
