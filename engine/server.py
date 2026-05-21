"""FastAPI server — computation backend called by Next.js API routes."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from engine.bias import NUMBER_POPULARITY, STAR_POPULARITY, uniqueness_score, score_tier
from engine.models import list_models, run_model, run_ensemble
from engine.data import (
    load_picks,
    save_pick,
    make_pick,
    load_draws,
    save_draws,
    compute_stats,
    parse_csv,
    best_draw_match,
)
from engine.generator import generate_picks
from engine.predictor import generate_predictions, top_signals, DISCLAIMER

app = FastAPI(title="EuroMillions Crowd Analyser", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/response models ───────────────────────────────────────────────────


class GenerateRequest(BaseModel):
    count: int = 1
    blendDraws: bool = False


class AnalyzeRequest(BaseModel):
    numbers: list[int]
    stars: list[int]


class PredictRequest(BaseModel):
    count: int = 3
    weights: dict[str, float] | None = None


class SavePickRequest(BaseModel):
    numbers: list[int]
    stars: list[int]
    uniquenessScore: int
    source: str = "manual"
    note: str | None = None


# ── Routes ────────────────────────────────────────────────────────────────────


@app.post("/generate")
def generate(req: GenerateRequest) -> list[dict]:
    if not 1 <= req.count <= 10:
        raise HTTPException(400, "count must be 1–10")
    return generate_picks(count=req.count, blend_draws=req.blendDraws)


@app.post("/analyze")
def analyze(req: AnalyzeRequest) -> dict:
    nums = req.numbers
    stars = req.stars

    if len(nums) != 5 or not all(1 <= n <= 50 for n in nums):
        raise HTTPException(400, "numbers must be 5 values in range 1–50")
    if len(stars) != 2 or not all(1 <= s <= 12 for s in stars):
        raise HTTPException(400, "stars must be 2 values in range 1–12")

    score = uniqueness_score(nums, stars)
    label, emoji = score_tier(score)

    breakdown = {
        "numbers": [
            {
                "number": n,
                "popularity": NUMBER_POPULARITY[n],
                "factors": _describe_number_factors(n),
            }
            for n in sorted(nums)
        ],
        "stars": [
            {
                "star": s,
                "popularity": STAR_POPULARITY[s],
                "factors": _describe_star_factors(s),
            }
            for s in sorted(stars)
        ],
    }

    return {
        "numbers": sorted(nums),
        "stars": sorted(stars),
        "uniquenessScore": score,
        "tier": label,
        "tierEmoji": emoji,
        "breakdown": breakdown,
    }


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    draws = load_draws()
    if len(draws) < 50:
        raise HTTPException(
            422,
            "At least 50 draws required. Import draw history first.",
        )
    try:
        predictions = generate_predictions(count=req.count, weights=req.weights)
    except ValueError as e:
        raise HTTPException(422, str(e))

    signals = top_signals(draws)
    return {
        "predictions": predictions,
        "signals": signals,
        "disclaimer": DISCLAIMER,
    }


@app.get("/picks")
def get_picks() -> list[dict]:
    picks = load_picks()
    draws = load_draws()
    for pick in picks:
        match = best_draw_match(pick, draws)
        if match:
            pick["bestMatch"] = {
                "numMatches": match["numMatches"],
                "starMatches": match["starMatches"],
                "date": match["draw"]["date"],
            }
    return picks


@app.post("/picks")
def post_pick(req: SavePickRequest) -> dict:
    if len(req.numbers) != 5 or not all(1 <= n <= 50 for n in req.numbers):
        raise HTTPException(400, "numbers must be 5 values in range 1–50")
    if len(req.stars) != 2 or not all(1 <= s <= 12 for s in req.stars):
        raise HTTPException(400, "stars must be 2 values in range 1–12")

    pick = make_pick(req.numbers, req.stars, req.uniquenessScore, req.source, req.note)
    save_pick(pick)
    return pick


@app.get("/draws")
def get_draws() -> list[dict]:
    return load_draws()


@app.post("/import")
async def import_csv(file: UploadFile = File(...)) -> dict:
    content = await file.read()
    try:
        draws = parse_csv(content)
    except ValueError as e:
        raise HTTPException(400, str(e))

    if not draws:
        raise HTTPException(400, "No valid draws found in CSV.")

    existing = load_draws()
    existing_dates = {d["date"] for d in existing}
    new_draws = [d for d in draws if d["date"] not in existing_dates]
    merged = sorted(existing + new_draws, key=lambda d: d["date"])
    save_draws(merged)
    stats = compute_stats(merged)

    return {
        "imported": len(new_draws),
        "total": len(merged),
        "dateRange": stats.get("dateRange", {}),
    }


@app.get("/stats")
def get_stats() -> dict:
    draws = load_draws()
    if not draws:
        return {}
    return compute_stats(draws)


@app.get("/accuracy")
def get_accuracy() -> dict:
    """Compare saved picks against subsequent draws for match distribution."""
    from engine.data import load_predictions
    picks = load_picks()
    draws = load_draws()

    if not picks or not draws:
        return {"distribution": [], "picks": 0, "draws": 0}

    draw_dates = {d["date"]: d for d in draws}
    dist: dict[str, int] = {}

    for pick in picks:
        pick_date = pick.get("timestamp", "")[:10]
        subsequent = [d for d in draws if d["date"] > pick_date]
        for draw in subsequent:
            nm = len(set(pick["numbers"]) & set(draw["numbers"]))
            sm = len(set(pick["stars"]) & set(draw["stars"]))
            key = f"{nm}n+{sm}s"
            dist[key] = dist.get(key, 0) + 1

    return {
        "distribution": [{"key": k, "count": v} for k, v in sorted(dist.items())],
        "picks": len(picks),
        "draws": len(draws),
    }


# ── ML Models ─────────────────────────────────────────────────────────────────


@app.get("/models")
def get_models() -> list[dict]:
    return list_models()


class RunModelRequest(BaseModel):
    model_id: str
    n_picks: int = 5
    n_samples: int = 300


class RunEnsembleRequest(BaseModel):
    model_ids: list[str]
    n_picks: int = 5
    n_samples: int = 300


@app.post("/models/run")
def run_single_model(req: RunModelRequest) -> dict:
    draws = load_draws()
    if not draws:
        raise HTTPException(422, "No draw history. Run fetch_draws.py --full first.")
    try:
        return run_model(req.model_id, n_picks=req.n_picks, n_samples=req.n_samples, draws=draws)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Model error: {e}")


@app.post("/models/ensemble")
def run_model_ensemble(req: RunEnsembleRequest) -> dict:
    draws = load_draws()
    if not draws:
        raise HTTPException(422, "No draw history. Run fetch_draws.py --full first.")
    try:
        return run_ensemble(req.model_ids, n_picks=req.n_picks, n_samples=req.n_samples, draws=draws)
    except Exception as e:
        raise HTTPException(500, f"Ensemble error: {e}")


# ── Factor description helpers ────────────────────────────────────────────────


def _describe_number_factors(n: int) -> list[str]:
    factors = []
    if 1 <= n <= 31:
        factors.append("Birthday range (+35)")
    if 1 <= n <= 12:
        factors.append("Month range (+15)")
    if 1 <= n <= 10:
        factors.append("Low number gravity (+18)")
    if n in {7, 17, 27, 37, 47}:
        factors.append("Lucky 7 family (+20)")
    if n == 7:
        factors.append("Lucky 7 apex (+15)")
    if n in {3, 13, 23, 33, 43}:
        factors.append("Lucky 3 family (+8)")
    if n in {10, 20, 30, 40, 50}:
        factors.append("Round number (+12)")
    if n in {1, 2, 3, 5, 8, 21, 34}:
        factors.append("Fibonacci (+12)")
    if 40 <= n <= 50:
        factors.append("High range neglect (-12)")
    if 45 <= n <= 50:
        factors.append("Very high neglect (-8)")
    if n == 13:
        factors.append("Unlucky 13 avoidance (-25)")
    return factors or ["No special bias"]


def _describe_star_factors(s: int) -> list[str]:
    factors = []
    if 1 <= s <= 6:
        factors.append("Low star bias (+30)")
    if 1 <= s <= 3:
        factors.append("Very low star (+20)")
    if s == 7:
        factors.append("Lucky star 7 (+15)")
    if 9 <= s <= 12:
        factors.append("High star neglect (-20)")
    if 11 <= s <= 12:
        factors.append("Very high star neglect (-12)")
    return factors or ["No special bias"]
