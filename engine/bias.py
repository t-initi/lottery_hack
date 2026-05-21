"""Human cognitive bias model for EuroMillions number popularity."""

from __future__ import annotations

BASELINE = 50
CLAMP_MIN = 10
CLAMP_MAX = 100

# ── Main numbers 1–50 ────────────────────────────────────────────────────────

_LUCKY_7_FAMILY = {7, 17, 27, 37, 47}
_LUCKY_3_FAMILY = {3, 13, 23, 33, 43}
_ROUND_NUMBERS = {10, 20, 30, 40, 50}
_FIBONACCI = {1, 2, 3, 5, 8, 21, 34}


def _number_popularity(n: int) -> int:
    score = BASELINE

    if 1 <= n <= 31:   # birthday effect
        score += 35
    if 1 <= n <= 12:   # month bonus
        score += 15
    if 1 <= n <= 10:   # low number gravity
        score += 18
    if n in _LUCKY_7_FAMILY:
        score += 20
    if n == 7:         # lucky 7 apex
        score += 15
    if n in _LUCKY_3_FAMILY:
        score += 8
    if n in _ROUND_NUMBERS:
        score += 12
    if n in _FIBONACCI:
        score += 12
    if 40 <= n <= 50:  # high neglect
        score -= 12
    if 45 <= n <= 50:  # very high neglect
        score -= 8
    if n == 13:        # unlucky 13 avoidance
        score -= 25

    return max(CLAMP_MIN, min(CLAMP_MAX, score))


NUMBER_POPULARITY: dict[int, int] = {n: _number_popularity(n) for n in range(1, 51)}

# ── Stars 1–12 ───────────────────────────────────────────────────────────────


def _star_popularity(s: int) -> int:
    score = BASELINE

    if 1 <= s <= 6:    # low star bias
        score += 30
    if 1 <= s <= 3:    # very low star
        score += 20
    if s == 7:         # lucky star 7
        score += 15
    if 9 <= s <= 12:   # high star neglect
        score -= 20
    if 11 <= s <= 12:  # very high star neglect
        score -= 12

    return max(CLAMP_MIN, min(CLAMP_MAX, score))


STAR_POPULARITY: dict[int, int] = {s: _star_popularity(s) for s in range(1, 13)}

# ── Uniqueness score ─────────────────────────────────────────────────────────


def uniqueness_score(numbers: list[int], stars: list[int]) -> int:
    """Return split-avoidance score 0–100. Higher = fewer expected splits."""
    pops = [NUMBER_POPULARITY[n] for n in numbers] + [STAR_POPULARITY[s] for s in stars]
    return round(100 - sum(pops) / len(pops))


# ── Tier labels ──────────────────────────────────────────────────────────────

def score_tier(score: int) -> tuple[str, str]:
    """Return (label, emoji) for a uniqueness score."""
    if score >= 85:
        return "Very Crowded", "🔴"
    if score >= 70:
        return "Crowded", "🟠"
    if score >= 50:
        return "Average", "⬜"
    if score >= 35:
        return "Underplayed", "🟢"
    return "Rare Pick", "🔵"


def pop_color(popularity: int) -> str:
    """CSS colour string for a popularity value, matching the design system tiers."""
    if popularity >= 85:
        return "#ff6b6b"   # red — very crowded
    if popularity >= 70:
        return "#ff9a3c"   # orange — crowded
    if popularity >= 50:
        return "#d0d8f0"   # text — average
    if popularity >= 35:
        return "#7ed4a4"   # green — underplayed
    return "#4fc3f7"       # cyan — rare
