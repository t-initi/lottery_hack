"""10-model ensemble prediction engine for EuroMillions."""

from __future__ import annotations

import random
import uuid
from collections import defaultdict
from datetime import datetime

from engine.bias import uniqueness_score
from engine.data import load_draws, save_prediction

DISCLAIMER = (
    "EuroMillions draws are independently certified as truly random using hardware RNG. "
    "No statistical model can predict lottery outcomes. This tool analyses human "
    "selection behaviour to minimise jackpot splits — not to improve winning odds. "
    "Track prediction accuracy over time to observe chance-level performance."
)

DEFAULT_WEIGHTS = {
    "frequency_regression": 0.20,
    "overdue": 0.20,
    "hot_streak": 0.10,
    "recency_decay": 0.25,
    "pair_frequency": 0.25,
}


# ── Individual model scorers ──────────────────────────────────────────────────


def _frequency_regression(draws: list[dict], window: int = 50) -> dict[int, float]:
    """Numbers drawn below expected frequency in last N draws."""
    recent = draws[-window:]
    freq: dict[int, int] = defaultdict(int)
    for d in recent:
        for n in d["numbers"]:
            freq[n] += 1
    expected = len(recent) * 5 / 50
    return {n: max(0.0, expected - freq.get(n, 0)) / expected for n in range(1, 51)}


def _overdue(draws: list[dict]) -> dict[int, float]:
    """Gap since last appearance, normalised to 20-draw max."""
    last_seen: dict[int, int] = {}
    for i, d in enumerate(draws):
        for n in d["numbers"]:
            last_seen[n] = i
    total = len(draws)
    scores = {}
    for n in range(1, 51):
        gap = total - last_seen.get(n, -1) - 1
        scores[n] = min(1.0, gap / 20.0)
    return scores


def _hot_streak(draws: list[dict], window: int = 20) -> dict[int, float]:
    """Numbers above expected frequency in last N draws."""
    recent = draws[-window:]
    freq: dict[int, int] = defaultdict(int)
    for d in recent:
        for n in d["numbers"]:
            freq[n] += 1
    expected = len(recent) * 5 / 50
    return {
        n: min(1.0, freq.get(n, 0) / max(1, expected * 2))
        for n in range(1, 51)
    }


def _sum_proximity(draws: list[dict]) -> dict[int, float]:
    """Score numbers that help a pick's sum approach the historical mean."""
    sums = [sum(d["numbers"]) for d in draws]
    target = sum(sums) / len(sums) if sums else 127.5
    # Proxy: prefer numbers close to target/5 (average contribution)
    avg = target / 5
    return {
        n: max(0.0, 1.0 - abs(n - avg) / avg)
        for n in range(1, 51)
    }


def _pair_frequency(draws: list[dict], window: int = 100) -> dict[int, float]:
    """Co-occurrence pair strength for each number."""
    recent = draws[-window:]
    pair_count: dict[frozenset, int] = defaultdict(int)
    for d in recent:
        nums = d["numbers"]
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                pair_count[frozenset([nums[i], nums[j]])] += 1

    num_score: dict[int, float] = defaultdict(float)
    for pair, cnt in pair_count.items():
        for n in pair:
            num_score[n] += cnt

    if not num_score:
        return {n: 0.5 for n in range(1, 51)}

    max_score = max(num_score.values())
    return {n: num_score.get(n, 0) / max_score for n in range(1, 51)}


def _recency_decay(draws: list[dict], lam: float = 0.92) -> dict[int, float]:
    """Exponential decay weight per draw back."""
    scores: dict[int, float] = defaultdict(float)
    total_weight = 0.0
    for i, d in enumerate(reversed(draws)):
        w = lam ** i
        total_weight += w
        for n in d["numbers"]:
            scores[n] += w
    if total_weight == 0:
        return {n: 0.5 for n in range(1, 51)}
    return {n: scores.get(n, 0) / (total_weight * 5 / 50) / 2 for n in range(1, 51)}


def _odd_even_balance(draws: list[dict], window: int = 50) -> dict[int, float]:
    """Prefer numbers matching historical odd/even ratio."""
    recent = draws[-window:]
    odd_count = sum(1 for d in recent for n in d["numbers"] if n % 2 == 1)
    total_nums = len(recent) * 5
    odd_ratio = odd_count / total_nums if total_nums else 0.5

    scores = {}
    for n in range(1, 51):
        if n % 2 == 1:
            scores[n] = odd_ratio
        else:
            scores[n] = 1.0 - odd_ratio
    return scores


def _decade_balance(draws: list[dict], window: int = 50) -> dict[int, float]:
    """Spread across decades 1-10, 11-20, 21-30, 31-40, 41-50."""
    recent = draws[-window:]
    decades = [0] * 5
    total = len(recent) * 5
    for d in recent:
        for n in d["numbers"]:
            decades[(n - 1) // 10] += 1
    target = total / 5
    decade_scores = [max(0.0, target - c) / target for c in decades]

    return {n: decade_scores[(n - 1) // 10] for n in range(1, 51)}


def _consecutive_gap(draws: list[dict], window: int = 50) -> dict[int, float]:
    """Prefer numbers that form historically-common consecutive patterns."""
    recent = draws[-window:]
    consecutive_freq: dict[int, int] = defaultdict(int)
    for d in recent:
        nums = sorted(d["numbers"])
        for i in range(len(nums) - 1):
            if nums[i + 1] - nums[i] == 1:
                consecutive_freq[nums[i]] += 1
                consecutive_freq[nums[i + 1]] += 1

    if not consecutive_freq:
        return {n: 0.5 for n in range(1, 51)}

    max_v = max(consecutive_freq.values())
    return {n: consecutive_freq.get(n, 0) / max_v for n in range(1, 51)}


def _star_overdue(draws: list[dict]) -> dict[int, float]:
    """Gap since each star last appeared."""
    last_seen: dict[int, int] = {}
    for i, d in enumerate(draws):
        for s in d["stars"]:
            last_seen[s] = i
    total = len(draws)
    scores = {}
    for s in range(1, 13):
        gap = total - last_seen.get(s, -1) - 1
        scores[s] = min(1.0, gap / 20.0)
    return scores


# ── Ensemble ──────────────────────────────────────────────────────────────────


def _run_models(draws: list[dict]) -> dict[str, dict[int, float]]:
    return {
        "frequency_regression": _frequency_regression(draws),
        "overdue": _overdue(draws),
        "hot_streak": _hot_streak(draws),
        "sum_proximity": _sum_proximity(draws),
        "pair_frequency": _pair_frequency(draws),
        "recency_decay": _recency_decay(draws),
        "odd_even_balance": _odd_even_balance(draws),
        "decade_balance": _decade_balance(draws),
        "consecutive_gap": _consecutive_gap(draws),
    }


def _ensemble(
    model_scores: dict[str, dict[int, float]],
    weights: dict[str, float],
) -> dict[int, float]:
    floor = 0.05
    combined: dict[int, float] = {}
    total_w = sum(weights.get(m, 0) for m in model_scores)

    for n in range(1, 51):
        score = floor
        for model, scores in model_scores.items():
            w = weights.get(model, 0)
            score += w * scores.get(n, 0)
        combined[n] = score / (1 + total_w) + floor

    return combined


def _pick_from_scores(
    num_scores: dict[int, float],
    star_scores: dict[int, float],
    target_sum: float,
    model_scores: dict[str, dict[int, float]],
    num_candidates: int = 20,
) -> dict:
    """Sample numbers by ensemble score, then filter for sum proximity."""

    def weighted_sample(scores: dict[int, float], k: int) -> list[int]:
        pop = list(scores.keys())
        wts = [scores[n] for n in pop]
        chosen = []
        rem_pop = list(pop)
        rem_wts = list(wts)
        for _ in range(k):
            total = sum(rem_wts)
            r = random.uniform(0, total)
            cum = 0.0
            idx = 0
            for i, w in enumerate(rem_wts):
                cum += w
                if r <= cum:
                    idx = i
                    break
            chosen.append(rem_pop[idx])
            rem_pop.pop(idx)
            rem_wts.pop(idx)
        return chosen

    best_pick = None
    best_dist = float("inf")

    for _ in range(num_candidates):
        nums = sorted(weighted_sample(num_scores, 5))
        dist = abs(sum(nums) - target_sum)
        if dist < best_dist:
            best_dist = dist
            best_pick = nums

    stars = sorted(weighted_sample(star_scores, 2))

    def reasons_for(n: int) -> list[str]:
        reasons = []
        for model, scores in model_scores.items():
            if scores.get(n, 0) > 0.5:
                reasons.append(model.replace("_", " "))
        return reasons or ["ensemble signal"]

    explanation = {
        "numbers": [
            {
                "number": n,
                "signal": round(num_scores.get(n, 0), 3),
                "reasons": reasons_for(n),
            }
            for n in best_pick
        ],
        "stars": [
            {
                "star": s,
                "signal": round(star_scores.get(s, 0), 3),
                "reasons": [],
            }
            for s in stars
        ],
    }

    score = uniqueness_score(best_pick, stars)
    return {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "numbers": best_pick,
        "stars": stars,
        "uniquenessScore": score,
        "source": "predicted",
        "sum": sum(best_pick),
        "sumTarget": round(target_sum, 1),
        "modelsUsed": list(model_scores.keys()),
        "explanation": explanation,
        "disclaimer": DISCLAIMER,
    }


def generate_predictions(
    count: int = 3,
    weights: dict[str, float] | None = None,
) -> list[dict]:
    draws = load_draws()
    if len(draws) < 50:
        raise ValueError("At least 50 draws required for predictions.")

    if weights is None:
        weights = DEFAULT_WEIGHTS

    model_scores = _run_models(draws)
    num_scores = _ensemble(model_scores, weights)

    star_ov = _star_overdue(draws)
    star_floor = 0.05
    star_scores = {s: star_ov[s] + star_floor for s in range(1, 13)}

    sums = [sum(d["numbers"]) for d in draws]
    target_sum = sum(sums) / len(sums)

    predictions = []
    for _ in range(count):
        pred = _pick_from_scores(num_scores, star_scores, target_sum, model_scores)
        save_prediction(pred)
        predictions.append(pred)

    return predictions


def top_signals(draws: list[dict] | None = None) -> dict:
    """Return top 10 number and 4 star signals for the predict tab."""
    if draws is None:
        draws = load_draws()
    if not draws:
        return {"numbers": [], "stars": []}

    model_scores = _run_models(draws)
    weights = DEFAULT_WEIGHTS
    num_scores = _ensemble(model_scores, weights)
    star_scores = _star_overdue(draws)

    top_nums = sorted(num_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    top_stars = sorted(star_scores.items(), key=lambda x: x[1], reverse=True)[:4]

    ov_scores = model_scores["overdue"]
    freq_scores = model_scores["frequency_regression"]
    rec_scores = model_scores["recency_decay"]
    hot_scores = model_scores["hot_streak"]

    return {
        "numbers": [
            {
                "number": n,
                "signal": round(s, 3),
                "overdue": round(ov_scores.get(n, 0), 3),
                "cold": round(freq_scores.get(n, 0), 3),
                "recency": round(rec_scores.get(n, 0), 3),
                "hot": round(hot_scores.get(n, 0), 3),
            }
            for n, s in top_nums
        ],
        "stars": [
            {"star": s, "signal": round(sc, 3)}
            for s, sc in top_stars
        ],
    }
