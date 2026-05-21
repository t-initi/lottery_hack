#!/usr/bin/env python3
"""
Fetch EuroMillions draw history from the FDJ (Française des Jeux) open data,
merge into data/draw_history.json, and update computed_stats.json.

FDJ publishes one zip per year:
  https://media.fdj.fr/static/csv/euromillions/euromillions_{YEAR}02.zip
The CSV inside uses semicolons and French column names
(boule_1…boule_5, etoile_1…etoile_2, date_de_tirage).

Smart caching:
  - Checks today's date against the last expected draw (Tuesday / Friday)
  - Skips the download if local data is already current
  - On a cache miss, downloads ONLY the current year's zip (~few KB)
  - Falls back to the UK XML endpoint for the very latest draw if the
    FDJ archive hasn't been updated yet

Usage:
    python scripts/fetch_draws.py               # normal run
    python scripts/fetch_draws.py --force       # skip staleness check
    python scripts/fetch_draws.py --full        # (re-)import all years from FDJ
    python scripts/fetch_draws.py --dry-run     # check only, no writes
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from datetime import date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from engine.data import DATA_DIR, load_draws, save_draws, compute_stats, parse_csv

# ── Config ────────────────────────────────────────────────────────────────────

UK_XML_URL = "https://www.national-lottery.co.uk/results/euromillions/draw-history/csv"
DRAW_DAYS  = {1, 4}   # Tuesday=1, Friday=4

# FDJ publishes historical draws in irregular "epoch" zip files.
# Only these codes currently resolve to valid archives.
# Each covers a different date range; together they span 2004-2024.
# The latest (202002) is the rolling file FDJ keeps updating (currently to Jul 2024).
FDJ_BASE = "https://media.fdj.fr/static/csv/euromillions/euromillions_{code}.zip"
FDJ_KNOWN_CODES = ["200402", "201402", "201902", "202002"]  # verified working

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
}


# ── Helpers ───────────────────────────────────────────────────────────────────


def last_expected_draw(today: date | None = None) -> date:
    d = today or date.today()
    for offset in range(7):
        candidate = d - timedelta(days=offset)
        if candidate.weekday() in DRAW_DAYS:
            return candidate
    return d


def is_up_to_date(draws: list[dict], reference: date) -> bool:
    if not draws:
        return False
    return max(d["date"] for d in draws) >= reference.isoformat()


def fetch_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def extract_csv_from_zip(raw: bytes) -> bytes:
    """Extract the first .csv file from a zip archive."""
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError("No CSV file found inside zip archive")
        return zf.read(csv_names[0])


def fetch_fdj_code(code: str) -> list[dict]:
    url = FDJ_BASE.format(code=code)
    print(f"  FDJ {code}: {url}", flush=True)
    raw_zip = fetch_bytes(url)
    csv_bytes = extract_csv_from_zip(raw_zip)
    draws = parse_csv(csv_bytes)
    print(f"    → {len(draws)} draws parsed")
    return draws


def fetch_uk_xml_draw() -> dict | None:
    """Fetch the latest draw from the UK XML endpoint as a fallback."""
    try:
        raw = fetch_bytes(UK_XML_URL)
        root = ET.fromstring(raw.decode("utf-8", errors="replace"))
        game = root.find(".//game[@type='euro']")
        if game is None:
            return None
        draw_date = (game.find(".//draw-date") or {}).text
        balls_el = game.find("balls")
        if not draw_date or balls_el is None:
            return None
        numbers = sorted(int(el.text) for el in balls_el.findall("ball") if el.text)
        stars = sorted(
            int(el.text)
            for el in balls_el.findall("bonus-ball")
            if el.get("type") == "luckystar" and el.text
        )
        if len(numbers) != 5 or len(stars) != 2:
            return None
        return {"date": draw_date.strip(), "numbers": numbers, "stars": stars}
    except Exception as e:
        print(f"  UK XML fallback failed: {e}", file=sys.stderr)
        return None


def merge(existing: list[dict], incoming: list[dict]) -> tuple[list[dict], int]:
    known = {d["date"] for d in existing}
    new = [d for d in incoming if d["date"] not in known]
    merged = sorted(existing + new, key=lambda d: d["date"])
    return merged, len(new)


def _warn_if_insufficient(draws: list[dict]) -> None:
    if len(draws) < 50:
        print(
            f"\n⚠  Only {len(draws)} draws locally — Predict tab needs 50+.\n"
            "   Run with --full to import all available years from FDJ."
        )


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch EuroMillions draws from FDJ")
    parser.add_argument("--force",   action="store_true", help="Download even if already up to date")
    parser.add_argument("--full",    action="store_true", help="Import all years from FDJ (initial load)")
    parser.add_argument("--dry-run", action="store_true", help="Check only, no writes")
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    existing = load_draws()
    expected = last_expected_draw()

    latest_local = existing[-1]["date"] if existing else "none"
    print(f"Last expected draw : {expected.isoformat()} ({expected.strftime('%A')})")
    print(f"Local draws        : {len(existing)} (latest: {latest_local})")

    # ── Full import ───────────────────────────────────────────────────────────
    if args.full:
        print(f"\nFull import: fetching {len(FDJ_KNOWN_CODES)} FDJ archive(s) …")
        all_draws: list[dict] = []
        for code in FDJ_KNOWN_CODES:
            try:
                all_draws.extend(fetch_fdj_code(code))
            except Exception as e:
                print(f"    ⚠ {code} failed: {e}")

        merged, n_new = merge(existing, all_draws)
        print(f"\n{n_new} new draws across all years (total: {len(merged)})")
        if args.dry_run:
            print("Dry run — no files written.")
            return 0
        if n_new:
            save_draws(merged)
            stats = compute_stats(merged)
            print(f"✓ Saved. Range: {stats['dateRange']['first']} → {stats['dateRange']['last']}")
        else:
            print("✓ Nothing new to add.")
        return 0

    # ── Staleness check ───────────────────────────────────────────────────────
    if not args.force and is_up_to_date(existing, expected):
        print("✓ Already up to date — nothing to download.")
        _warn_if_insufficient(existing)
        return 0

    # ── Incremental update: re-fetch the rolling FDJ archive (latest epoch) ──
    new_draws: list[dict] = []
    latest_code = FDJ_KNOWN_CODES[-1]

    try:
        new_draws = fetch_fdj_code(latest_code)
    except Exception as e:
        print(f"  FDJ {latest_code} failed: {e}")

    merged, n_new = merge(existing, new_draws)

    # ── FDJ might lag a few hours — try UK XML for the very latest draw ───────
    if not is_up_to_date(merged, expected):
        print("  FDJ archive may not include today's draw yet, trying UK fallback …")
        latest = fetch_uk_xml_draw()
        if latest:
            print(f"  UK fallback: {latest['date']} — {latest['numbers']} ★ {latest['stars']}")
            merged, extra = merge(merged, [latest])
            n_new += extra

    print(f"  {n_new} new draw(s) found")

    if args.dry_run:
        print("Dry run — no files written.")
        _warn_if_insufficient(merged)
        return 0

    if n_new == 0:
        print("✓ Nothing new to add.")
        _warn_if_insufficient(existing)
        return 0

    save_draws(merged)
    stats = compute_stats(merged)
    print(
        f"✓ Saved. Total: {len(merged)} draws "
        f"({stats['dateRange']['first']} → {stats['dateRange']['last']})"
    )
    _warn_if_insufficient(merged)
    return 0


if __name__ == "__main__":
    sys.exit(main())
