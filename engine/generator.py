"""Weighted pick generation using inverse-popularity bias."""

from __future__ import annotations

import random

from engine.bias import NUMBER_POPULARITY, STAR_POPULARITY, uniqueness_score
from engine.data import make_pick, save_pick, load_draws, compute_stats


def _blend_weights(
    weights: dict[int, float],
    freq: dict[str, int],
    expected: float,
    max_delta: float = 10.0,
) -> dict[int, float]:
    """Reduce weight for cold numbers by up to max_delta popularity points."""
    blended = {}
    for n, w in weights.items():
        observed = freq.get(str(n), 0)
        delta = 0.0
        if expected > 0:
            ratio = observed / expected
            if ratio < 1.0:
                delta = (1.0 - ratio) * max_delta
        blended[n] = max(1e-6, w - delta / 100.0)
    return blended


def _inverse_weights(popularity: dict[int, int]) -> dict[int, float]:
    return {n: 1.0 / p for n, p in popularity.items()}


def generate_picks(count: int = 1, blend_draws: bool = False) -> list[dict]:
    num_weights = _inverse_weights(NUMBER_POPULARITY)
    star_weights = _inverse_weights(STAR_POPULARITY)

    if blend_draws:
        draws = load_draws()
        if draws:
            stats = compute_stats(draws)
            num_weights = _blend_weights(
                num_weights,
                stats.get("numFrequency", {}),
                stats.get("expectedNumFreq", 0),
            )
            star_weights = _blend_weights(
                star_weights,
                stats.get("starFrequency", {}),
                stats.get("expectedStarFreq", 0),
            )

    picks = []
    for _ in range(count):
        numbers = _weighted_sample(num_weights, k=5)
        stars = _weighted_sample(star_weights, k=2)
        score = uniqueness_score(numbers, stars)
        pick = make_pick(sorted(numbers), sorted(stars), score, source="generated")
        save_pick(pick)
        picks.append(pick)

    return picks


def _weighted_sample(weights: dict[int, float], k: int) -> list[int]:
    population = list(weights.keys())
    wts = [weights[n] for n in population]
    chosen = []
    remaining_pop = list(population)
    remaining_wts = list(wts)

    for _ in range(k):
        total = sum(remaining_wts)
        r = random.uniform(0, total)
        cumulative = 0.0
        idx = 0
        for i, w in enumerate(remaining_wts):
            cumulative += w
            if r <= cumulative:
                idx = i
                break
        chosen.append(remaining_pop[idx])
        remaining_pop.pop(idx)
        remaining_wts.pop(idx)

    return chosen
