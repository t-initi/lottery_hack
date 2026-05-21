"""Draw/pick schema, persistence, CSV parser, and statistics."""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, date
from pathlib import Path
from typing import Any

import pandas as pd

from engine.bias import NUMBER_POPULARITY, STAR_POPULARITY

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

PICKS_FILE = DATA_DIR / "my_picks.json"
DRAWS_FILE = DATA_DIR / "draw_history.json"
STATS_FILE = DATA_DIR / "computed_stats.json"
PREDICTIONS_FILE = DATA_DIR / "predictions.json"

# ── Schema helpers ────────────────────────────────────────────────────────────


def make_pick(
    numbers: list[int],
    stars: list[int],
    uniqueness: int,
    source: str = "manual",
    note: str | None = None,
) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "numbers": sorted(numbers),
        "stars": sorted(stars),
        "uniquenessScore": uniqueness,
        "source": source,
        "note": note,
    }


# ── Persistence ───────────────────────────────────────────────────────────────


def _load(path: Path, default: Any = None):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return default if default is not None else []


def _save(path: Path, data: Any) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_picks() -> list[dict]:
    return _load(PICKS_FILE, [])


def save_pick(pick: dict) -> None:
    picks = load_picks()
    picks.append(pick)
    _save(PICKS_FILE, picks)


def load_draws() -> list[dict]:
    return _load(DRAWS_FILE, [])


def save_draws(draws: list[dict]) -> None:
    _save(DRAWS_FILE, draws)


def load_predictions() -> list[dict]:
    return _load(PREDICTIONS_FILE, [])


def save_prediction(pred: dict) -> None:
    preds = load_predictions()
    preds.append(pred)
    _save(PREDICTIONS_FILE, preds)


# ── CSV Parser ────────────────────────────────────────────────────────────────

# Maps flexible column aliases → canonical names
# Includes FDJ French format: boule_1…boule_5, etoile_1…etoile_2, date_de_tirage
_NUM_ALIASES = [
    ["ball 1", "ball1", "n1", "number1", "num1", "boule 1", "boule1"],
    ["ball 2", "ball2", "n2", "number2", "num2", "boule 2", "boule2"],
    ["ball 3", "ball3", "n3", "number3", "num3", "boule 3", "boule3"],
    ["ball 4", "ball4", "n4", "number4", "num4", "boule 4", "boule4"],
    ["ball 5", "ball5", "n5", "number5", "num5", "boule 5", "boule5"],
]
_STAR_ALIASES = [
    ["lucky star 1", "luckystar1", "star1", "s1", "lucky1", "etoile 1", "etoile1"],
    ["lucky star 2", "luckystar2", "star2", "s2", "lucky2", "etoile 2", "etoile2"],
]
_DATE_ALIASES = ["draw date", "drawdate", "date", "draw_date", "date de tirage", "date de tirage"]


def _normalise(col: str) -> str:
    return re.sub(r"[\s_-]+", " ", col.strip().lower())


def _find_col(columns: list[str], aliases: list[str]) -> str | None:
    col_map = {_normalise(c): c for c in columns}
    for alias in aliases:
        if alias in col_map:
            return col_map[alias]
    return None


def parse_csv(content: bytes | str) -> list[dict]:
    """Parse a EuroMillions CSV (various column formats) into Draw dicts."""
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="replace")

    df = pd.read_csv(pd.io.common.StringIO(content), sep=None, engine="python")
    df.columns = df.columns.str.strip()

    cols = df.columns.tolist()

    date_col = _find_col(cols, _DATE_ALIASES)
    num_cols = [_find_col(cols, a) for a in _NUM_ALIASES]
    star_cols = [_find_col(cols, a) for a in _STAR_ALIASES]

    if not date_col or any(c is None for c in num_cols) or any(c is None for c in star_cols):
        raise ValueError(
            f"Cannot identify required columns in CSV. "
            f"Found: {cols}. Expected date, 5 ball, 2 lucky-star columns."
        )

    draws: list[dict] = []
    for _, row in df.iterrows():
        try:
            raw_date = str(row[date_col]).strip()
            # Try ISO format first (YYYY-MM-DD or YYYYMMDD), then DD/MM/YYYY
            for fmt in ("%Y-%m-%d", "%Y%m%d", "%d/%m/%Y", "%d-%m-%Y"):
                try:
                    parsed = pd.Timestamp(raw_date) if fmt in ("%Y-%m-%d",) else pd.to_datetime(raw_date, format=fmt)
                    break
                except Exception:
                    continue
            else:
                parsed = pd.to_datetime(raw_date, dayfirst=True)
            iso_date = parsed.strftime("%Y-%m-%d")

            numbers = sorted(int(row[c]) for c in num_cols)
            stars = sorted(int(row[c]) for c in star_cols)

            if not (len(numbers) == 5 and all(1 <= n <= 50 for n in numbers)):
                continue
            if not (len(stars) == 2 and all(1 <= s <= 12 for s in stars)):
                continue

            draw: dict = {"date": iso_date, "numbers": numbers, "stars": stars}

            for jcol in ["jackpot", "jackpot_eur", "prize"]:
                if jcol in df.columns:
                    try:
                        draw["jackpot"] = float(str(row[jcol]).replace(",", "").replace("€", ""))
                    except Exception:
                        pass
                    break

            for wcol in ["winners", "jackpot_winners", "num_winners"]:
                if wcol in df.columns:
                    try:
                        draw["winners"] = int(row[wcol])
                    except Exception:
                        pass
                    break

            draws.append(draw)
        except Exception:
            continue

    draws.sort(key=lambda d: d["date"])
    return draws


# ── Statistics ────────────────────────────────────────────────────────────────


def compute_stats(draws: list[dict]) -> dict:
    if not draws:
        return {}

    dates = [d["date"] for d in draws]
    total = len(draws)

    num_freq: dict[int, int] = {n: 0 for n in range(1, 51)}
    star_freq: dict[int, int] = {s: 0 for s in range(1, 13)}
    sums: list[int] = []
    pair_counts: dict[str, int] = {}
    star_pair_counts: dict[str, int] = {}

    for draw in draws:
        nums = draw["numbers"]
        stars = draw["stars"]
        for n in nums:
            num_freq[n] += 1
        for s in stars:
            star_freq[s] += 1
        sums.append(sum(nums))

        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                key = f"{nums[i]},{nums[j]}"
                pair_counts[key] = pair_counts.get(key, 0) + 1

        key2 = f"{stars[0]},{stars[1]}"
        star_pair_counts[key2] = star_pair_counts.get(key2, 0) + 1

    expected_num = total * 5 / 50
    expected_star = total * 2 / 12

    sorted_nums = sorted(num_freq.items(), key=lambda x: x[1], reverse=True)
    hot_numbers = [n for n, _ in sorted_nums[:10]]
    cold_numbers = [n for n, _ in sorted_nums[-10:]]

    sorted_stars = sorted(star_freq.items(), key=lambda x: x[1], reverse=True)
    hot_stars = [s for s, _ in sorted_stars[:4]]
    cold_stars = [s for s, _ in sorted_stars[-4:]]

    top_pairs = [
        {"pair": [int(k.split(",")[0]), int(k.split(",")[1])], "count": v}
        for k, v in sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    ]
    top_star_pairs = [
        {"pair": [int(k.split(",")[0]), int(k.split(",")[1])], "count": v}
        for k, v in sorted(star_pair_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    ]

    sums_arr = pd.Series(sums)
    stats = {
        "totalDraws": total,
        "dateRange": {"first": min(dates), "last": max(dates)},
        "numFrequency": {str(k): v for k, v in num_freq.items()},
        "starFrequency": {str(k): v for k, v in star_freq.items()},
        "expectedNumFreq": round(expected_num, 2),
        "expectedStarFreq": round(expected_star, 2),
        "hotNumbers": hot_numbers,
        "coldNumbers": cold_numbers,
        "hotStars": hot_stars,
        "coldStars": cold_stars,
        "topPairs": top_pairs,
        "topStarPairs": top_star_pairs,
        "sumStats": {
            "mean": round(float(sums_arr.mean()), 2),
            "median": round(float(sums_arr.median()), 2),
            "std": round(float(sums_arr.std()), 2),
            "min": int(sums_arr.min()),
            "max": int(sums_arr.max()),
        },
    }

    _save(STATS_FILE, stats)
    return stats


def load_stats() -> dict:
    return _load(STATS_FILE, {})


# ── Match scoring ─────────────────────────────────────────────────────────────


def best_draw_match(pick: dict, draws: list[dict]) -> dict | None:
    """Find the draw with the most overlapping numbers+stars for a pick."""
    if not draws:
        return None

    best = None
    best_score = -1

    pick_nums = set(pick["numbers"])
    pick_stars = set(pick["stars"])

    for draw in draws:
        nm = len(pick_nums & set(draw["numbers"]))
        sm = len(pick_stars & set(draw["stars"]))
        score = nm * 10 + sm
        if score > best_score:
            best_score = score
            best = {"draw": draw, "numMatches": nm, "starMatches": sm}

    return best
