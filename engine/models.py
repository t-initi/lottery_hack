"""
Unified runner for the trained ML models in ../jupyter-scripts/euromillions/.

All models were trained on `lottery_data.csv` — 5 columns of main numbers,
no stars, no date. Each model predicts the next 5 numbers given recent history.

Model zoo:
  lstm   — Keras LSTM (lstm_lottery_model_new_11.keras) + lstm_scalers.pkl
  rf     — RandomForest (rf_model2.pkl)
  xgb    — XGBoost list of 5 regressors (xgboost_model2.pkl) + scaler.pkl
  nn     — Neural network (nn_model.pkl)
  linear — Linear model (linear_model.pkl)

Each runner returns a list of dicts:
  [{"numbers": [int, ...], "confidence": float, "sample_index": int}, ...]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from engine.bias import STAR_POPULARITY, uniqueness_score
from engine.data import load_draws

# ── Path to the trained model artefacts ──────────────────────────────────────

MODELS_DIR = Path(__file__).parent.parent.parent / "jupyter-scripts" / "euromillions"

# ── Registry ──────────────────────────────────────────────────────────────────

_REGISTRY: dict[str, dict] = {
    "lstm": {
        "label":       "LSTM (deep sequence model)",
        "model_file":  "lstm_lottery_model_new_11.keras",
        "scaler_file": "lstm_scalers.pkl",
        "description": "Long short-term memory neural network trained on sequences of 10 draws.",
        "sequence_len": 10,
    },
    "rf": {
        "label":       "Random Forest",
        "model_file":  "rf_model2.pkl",
        "scaler_file": None,
        "description": "Ensemble of 200 decision trees, trained on next-row prediction.",
    },
    "xgb": {
        "label":       "XGBoost",
        "model_file":  "xgboost_model2.pkl",
        "scaler_file": "scaler.pkl",
        "description": "Gradient boosting — 5 separate regressors, one per number position.",
    },
    "nn": {
        "label":       "Neural Network",
        "model_file":  "nn_model.pkl",
        "scaler_file": None,
        "description": "Scikit-learn MLP trained on next-row prediction.",
    },
    "linear": {
        "label":       "Linear Model",
        "model_file":  "linear_model.pkl",
        "scaler_file": None,
        "description": "Linear regression baseline — minimal assumptions.",
    },
}


def _tensorflow_available() -> bool:
    try:
        import tensorflow  # noqa: F401
        return True
    except ImportError:
        return False


def list_models() -> list[dict]:
    """Return available model metadata, marking which files are present."""
    tf_ok = None  # lazy-check only once per call

    result = []
    for key, meta in _REGISTRY.items():
        model_path = MODELS_DIR / meta["model_file"]
        scaler_path = (MODELS_DIR / meta["scaler_file"]) if meta.get("scaler_file") else None
        available = model_path.exists() and (scaler_path is None or scaler_path.exists())
        reason: str | None = None

        if not model_path.exists():
            reason = f"model file not found: {meta['model_file']}"
        elif scaler_path and not scaler_path.exists():
            reason = f"scaler file not found: {meta['scaler_file']}"
        elif key == "lstm" and available:
            if tf_ok is None:
                tf_ok = _tensorflow_available()
            if not tf_ok:
                available = False
                reason = "tensorflow not installed — run: pip install tensorflow"

        entry = {
            "id":          key,
            "label":       meta["label"],
            "description": meta["description"],
            "available":   available,
            "model_file":  meta["model_file"],
        }
        if reason:
            entry["reason"] = reason
        result.append(entry)
    return result


# ── Data preparation ──────────────────────────────────────────────────────────


def _draws_to_matrix(draws: list[dict]) -> np.ndarray:
    """Convert draw dicts to (N, 5) float32 matrix of main numbers."""
    return np.array([d["numbers"] for d in draws], dtype=np.float32)


def _clamp_numbers(raw: np.ndarray) -> list[int]:
    """Round and clamp to valid EuroMillions range, deduplicate."""
    nums = sorted(int(round(n)) for n in raw)
    clamped = [max(1, min(50, n)) for n in nums]
    # Deduplicate while preserving as many as possible
    seen: set[int] = set()
    result: list[int] = []
    for n in sorted(clamped):
        while n in seen and n < 50:
            n += 1
        if n not in seen:
            seen.add(n)
            result.append(n)
    return sorted(result[:5])


def _sample_stars(rng: np.random.Generator | None = None) -> list[int]:
    """
    None of the trained models predict stars. Sample 2 stars weighted
    inversely by STAR_POPULARITY so we stay consistent with the bias model.
    """
    r = rng or np.random.default_rng()
    pop = np.array([STAR_POPULARITY[s] for s in range(1, 13)], dtype=float)
    weights = 1.0 / pop
    weights /= weights.sum()
    return sorted(int(s) for s in r.choice(range(1, 13), size=2, replace=False, p=weights))


# ── Per-model runners ─────────────────────────────────────────────────────────


def _run_lstm(data: np.ndarray, n_picks: int, n_samples: int) -> list[dict]:
    import joblib
    meta = _REGISTRY["lstm"]
    seq_len = meta["sequence_len"]

    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError:
        raise RuntimeError(
            "TensorFlow is not installed in the active Python environment. "
            "Fix: activate the venv and run `pip install tensorflow`"
        )

    model = keras.models.load_model(
        str(MODELS_DIR / meta["model_file"]),
        custom_objects={"mse": tf.keras.losses.MeanSquaredError()},
        compile=False,
    )
    scalers = joblib.load(MODELS_DIR / meta["scaler_file"])

    # Normalise each column
    norm = np.zeros_like(data, dtype=float)
    for i in range(data.shape[1]):
        norm[:, i] = scalers[i].transform(data[:, i].reshape(-1, 1)).flatten()

    seq = norm[-seq_len:].reshape(1, seq_len, data.shape[1])

    rng = np.random.default_rng(42)
    picks: list[dict] = []
    seen: set[tuple] = set()

    for _ in range(n_samples):
        pred_norm = model.predict(seq, verbose=0)[0]
        # Inverse-transform each column
        pred_raw = np.array([
            scalers[i].inverse_transform([[pred_norm[i]]])[0][0]
            for i in range(len(scalers))
        ])
        # Add small noise for diversity
        noisy = pred_raw + rng.normal(0, 1.5, size=pred_raw.shape)
        nums = tuple(_clamp_numbers(noisy))
        if len(nums) == 5 and nums not in seen:
            seen.add(nums)
            picks.append({"numbers": list(nums), "sample_index": len(picks)})
        if len(picks) >= n_picks:
            break

    return picks[:n_picks]


def _run_sklearn_model(model_obj: Any, data: np.ndarray, n_picks: int, n_samples: int,
                       scaler=None) -> list[dict]:
    """Generic runner for RF / NN / Linear (all share the same sklearn interface)."""
    rng = np.random.default_rng(42)

    X = data
    if scaler is not None:
        # XGBoost expects frequency features appended
        flat = data.flatten()
        unique, counts = np.unique(flat, return_counts=True)
        freq_dict = dict(zip(unique.astype(int), counts))
        freq_features = np.array([[freq_dict.get(int(n), 0) for n in row] for row in data])
        X_aug = np.hstack([data, freq_features])
        X = scaler.transform(X_aug)

    last_row = X[-1].reshape(1, -1)

    picks: list[dict] = []
    seen: set[tuple] = set()

    for _ in range(n_samples):
        if isinstance(model_obj, list):
            # XGBoost: list of 5 models
            raw = np.array([float(m.predict(last_row)[0]) for m in model_obj])
        else:
            raw = model_obj.predict(last_row)[0]

        noise = rng.normal(0, 2.0, size=5)
        nums = tuple(_clamp_numbers(raw + noise))
        if len(nums) == 5 and nums not in seen:
            seen.add(nums)
            picks.append({"numbers": list(nums), "sample_index": len(picks)})
        if len(picks) >= n_picks:
            break

    return picks[:n_picks]


# ── Public API ────────────────────────────────────────────────────────────────


def run_model(
    model_id: str,
    n_picks: int = 5,
    n_samples: int = 500,
    draws: list[dict] | None = None,
) -> dict:
    """
    Run a named model and return structured predictions.

    Returns:
      {
        "model_id": str,
        "label": str,
        "picks": [{"numbers": [int, ...], "stars": [int, int], "uniquenessScore": int}],
        "frequency": {1: count, 2: count, ...}   # across all Monte Carlo samples
      }
    """
    if model_id not in _REGISTRY:
        raise ValueError(f"Unknown model '{model_id}'. Available: {list(_REGISTRY)}")

    meta = _REGISTRY[model_id]
    model_path = MODELS_DIR / meta["model_file"]
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if draws is None:
        draws = load_draws()
    if not draws:
        raise ValueError("No draw history loaded. Run fetch_draws.py --full first.")

    data = _draws_to_matrix(draws)

    import joblib

    rng = np.random.default_rng(42)

    if model_id == "lstm":
        raw_picks = _run_lstm(data, n_picks, n_samples)

    elif model_id in ("rf", "nn", "linear"):
        model_obj = joblib.load(model_path)
        raw_picks = _run_sklearn_model(model_obj, data, n_picks, n_samples)

    elif model_id == "xgb":
        scaler_path = MODELS_DIR / meta["scaler_file"]
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        model_obj = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        raw_picks = _run_sklearn_model(model_obj, data, n_picks, n_samples, scaler=scaler)

    else:
        raise ValueError(f"No runner implemented for '{model_id}'")

    # Count per-number frequency across all picks for the consensus chart
    freq: dict[int, int] = {n: 0 for n in range(1, 51)}
    for p in raw_picks:
        for n in p["numbers"]:
            freq[n] += 1

    # Build final picks with stars + uniqueness
    picks = []
    for p in raw_picks:
        stars = _sample_stars(rng)
        score = uniqueness_score(p["numbers"], stars)
        picks.append({
            "numbers":        p["numbers"],
            "stars":          stars,
            "uniquenessScore": score,
        })

    return {
        "model_id":  model_id,
        "label":     meta["label"],
        "picks":     picks,
        "frequency": {str(k): v for k, v in freq.items()},
    }


def run_ensemble(
    model_ids: list[str],
    n_picks: int = 5,
    n_samples: int = 300,
    draws: list[dict] | None = None,
) -> dict:
    """
    Run multiple models and aggregate their Monte Carlo frequency distributions.

    Returns per-model results PLUS a consensus frequency map showing how many
    models "voted" for each number.
    """
    if draws is None:
        draws = load_draws()

    results: list[dict] = []
    consensus: dict[int, float] = {n: 0.0 for n in range(1, 51)}

    for mid in model_ids:
        try:
            res = run_model(mid, n_picks=n_picks, n_samples=n_samples, draws=draws)
            results.append(res)
            total = sum(res["frequency"].values()) or 1
            for n in range(1, 51):
                consensus[n] += res["frequency"].get(str(n), 0) / total
        except Exception as e:
            results.append({"model_id": mid, "error": str(e), "picks": [], "frequency": {}})

    # Normalise consensus to [0, 1]
    max_v = max(consensus.values()) or 1
    consensus_norm = {str(k): round(v / max_v, 4) for k, v in consensus.items()}

    # Top 10 consensus numbers
    top10 = sorted(
        range(1, 51), key=lambda n: consensus_norm[str(n)], reverse=True
    )[:10]

    return {
        "models":          results,
        "consensus":       consensus_norm,
        "consensusTop10":  top10,
    }
