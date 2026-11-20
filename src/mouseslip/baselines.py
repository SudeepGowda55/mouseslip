"""Honest baselines and the ablation: what the two clocks add over what is known before any move is timed."""

import json
import pickle
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mouseslip.data import DATASET, Window, load_windows
from mouseslip.predict import BASELINE_MODEL, ROOT

RESULTS = ROOT / "results" / "baselines.json"
RECENT = 5  # own moves, the horizon the dataset's rationales use
MOVES_CAP = 200.0  # "the clock lasts N more moves" stops meaning anything beyond this


def time_control(window: Window) -> tuple[float, float]:
    """Return (base, increment) in seconds."""
    base, increment = window.time_control.split("+")
    return float(base), float(increment)


def speed_bucket(window: Window) -> str:
    """Name the speed class from the estimated game length, base + 40 increments."""
    base, increment = time_control(window)
    length = base + 40 * increment
    return "bullet" if length < 180 else "blitz" if length < 480 else "rapid" if length < 1500 else "classical"  # noqa: PLR2004


def move_times(window: Window) -> tuple[np.ndarray, np.ndarray]:
    """Return the seconds the mover and the opponent spent on each of their own moves in the window, oldest first."""
    plies = len(window.white_clock)
    base, increment = time_control(window)
    mine, theirs = window.white_clock, window.black_clock
    if window.side_to_move == "black":
        mine, theirs = theirs, mine
    spent = []
    # the mover's last move is two plies back and the opponent's is one, so each side's own readings sit on one parity
    for readings in (mine[plies % 2 :: 2].astype(float), theirs[(plies + 1) % 2 :: 2].astype(float)):
        # a berserker gives up the increment along with half the clock
        # shortcut: only visible while the window still holds the game's first move; past ply 40 a berserker's move
        # times read one increment too long, and moves_left always subtracts the increment. Under 1% of increment
        # windows; fix by passing the berserk flag through Window if that ever matters.
        # the half second covers whole-second rounding in the dataset and the lag credit on a live first move
        berserk = window.ply == plies and len(readings) and readings[0] <= base / 2 + 0.5
        spent.append(np.clip(readings[:-1] + (0 if berserk else increment) - readings[1:], 0, None))
    return spent[0], spent[1]


def moves_left(left: float, spent: np.ndarray, increment: float) -> float:
    """Return how many more moves the clock lasts at the recent pace; infinite when the increment covers it."""
    burn = spent[-RECENT:].mean() - increment if len(spent) else 0.0
    return left / burn if burn > 0 else float("inf")


def priors(window: Window) -> dict[str, float]:
    """What is known before a single move is timed."""
    base, increment = time_control(window)
    white = window.side_to_move == "white"
    mover, opponent = (window.white_elo, window.black_elo) if white else (window.black_elo, window.white_elo)
    return {
        "mover_elo": mover,
        "elo_gap": mover - opponent,
        "log_base": np.log1p(base),
        "increment": increment,
        "ply": window.ply,
        "white": float(white),
    }


def _recent(name: str, spent: np.ndarray) -> dict[str, float]:
    last = spent[-RECENT:] if len(spent) else np.zeros(1)
    return {f"{name}_mean": last.mean(), f"{name}_max": last.max(), f"{name}_last": last[-1], f"{name}_std": last.std()}


def clock_dynamics(window: Window) -> dict[str, float]:
    """What the two clocks alone say: no rating, no ply. Seconds left still give the time control away."""
    base, increment = time_control(window)
    length = base + 40 * increment
    white = window.side_to_move == "white"
    left, opponent_left = (float(c[-1]) for c in (window.white_clock, window.black_clock)[:: 1 if white else -1])
    own, opponent = move_times(window)
    every = own if len(own) else np.zeros(1)
    return {
        "left": left,
        "opponent_left": opponent_left,
        "left_share": left / length,
        "opponent_left_share": opponent_left / length,
        "left_gap_share": (left - opponent_left) / length,
        **_recent("own", own),
        **_recent("opponent", opponent),
        "moves_left": min(moves_left(left, own, increment), MOVES_CAP),
        "opponent_moves_left": min(moves_left(opponent_left, opponent, increment), MOVES_CAP),
        "trend": every[-RECENT:].mean() - every[:-RECENT].mean() if len(every) > RECENT else 0.0,
        "longest": every.max(),
        "longest_share": every.max() / length,
        "recent_share_of_left": every[-RECENT:].mean() / (left + 1),
        "instant_moves": float((every[-RECENT:] <= 1).mean()),
    }


def engine(window: Window) -> dict[str, float]:
    """The evaluation from the mover's view. Offline only, and needs the dataset's minimum of 12 plies."""
    pawns = window.evaluation.astype(float) * (1 if window.side_to_move == "white" else -1)
    chances = 2 / (1 + np.exp(-0.368208 * pawns)) - 1  # Lichess winning chances, the scale blunders are defined on
    recent, steps = pawns[-10:], np.diff(chances[-11:])
    return {
        "eval": pawns[-1],
        "eval_abs": abs(pawns[-1]),
        "chances": chances[-1],
        "eval_swing": recent.max() - recent.min(),
        "eval_volatility": np.diff(recent).std(),
        "eval_change_1": pawns[-1] - pawns[-2],
        "eval_change_2": pawns[-1] - pawns[-3],
        "own_worst_drop": max(0.0, -steps[-2::-2].min()),  # the last step is the opponent's move
        "opponent_worst_drop": max(0.0, steps[::-2].max()),
    }


FEATURE_SETS = {
    "priors": (priors,),
    "clock-dynamics": (clock_dynamics,),
    "clocks": (priors, clock_dynamics),
    "clocks+engine": (priors, clock_dynamics, engine),
}


def features(window: Window, feature_set: str) -> dict[str, float]:
    """Return one window's named features for a feature set."""
    return {name: value for part in FEATURE_SETS[feature_set] for name, value in part(window).items()}


def matrix(windows: list[Window], feature_set: str) -> np.ndarray:
    """Return the feature matrix, one row per window."""
    return np.array([list(features(w, feature_set).values()) for w in windows], dtype=np.float32)


class BaselinePredictor:
    """The "clocks" gradient-boosting model behind the Predictor protocol."""

    name = "gradient boosting on clocks, ratings and time control"

    def __init__(self, path: Path | str = BASELINE_MODEL) -> None:
        saved = pickle.loads(Path(path).read_bytes())  # noqa: S301 (our own file)
        self.model = saved["model"]
        self.threshold: float = saved["threshold"]  # marks the top decile of validation scores

    def p_blunder(self, window: Window) -> float:
        return float(self.model.predict_proba(matrix([window], "clocks"))[0, 1])

    def explain(self, window: Window) -> str:
        _, increment = time_control(window)
        white = window.side_to_move == "white"
        left, opponent_left = (float(c[-1]) for c in (window.white_clock, window.black_clock)[:: 1 if white else -1])
        text = f"{window.side_to_move.capitalize()} has {_clock(left)} left against {_clock(opponent_left)}."
        recent = move_times(window)[0][-RECENT:]
        if not len(recent):
            return text
        lasts = moves_left(left, recent, increment)
        took = f"last {len(recent)} moves took {recent.mean():.1f} seconds each" if len(recent) > 1 else f"last move took {recent[0]:.1f} seconds"
        pace = f"so the clock lasts about {lasts:.0f} more move{'s' if round(lasts) != 1 else ''} at this pace" if lasts < MOVES_CAP else "so the clock is not a worry at this pace"
        return f"{text} Their {took}, {pace}."


def _clock(seconds: float) -> str:
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"


def fit_logistic(x, y, x_val, y_val):
    """Return (model, parameters, validation AUC) for the best C on validation."""
    fits = []
    for c in (0.001, 0.01, 0.1, 1.0, 10.0):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000)).fit(x, y)
        fits.append((roc_auc_score(y_val, model.predict_proba(x_val)[:, 1]), {"C": c}, model))
    auc, params, model = max(fits, key=lambda fit: fit[0])
    return model, params, auc


def fit_boosting(x, y, x_val, y_val):
    """Return (model, parameters, validation AUC); the grid and the stopping round both look at validation only."""
    fits = []
    for rate, leaves, leaf in product((0.03, 0.1), (4, 8, 16), (20, 200)):
        params = {"learning_rate": rate, "max_leaf_nodes": leaves, "min_samples_leaf": leaf}
        model = HistGradientBoostingClassifier(max_iter=1000, early_stopping=True, n_iter_no_change=30, random_state=0, **params)
        model.fit(x, y, X_val=x_val, y_val=y_val)
        fits.append((roc_auc_score(y_val, model.predict_proba(x_val)[:, 1]), params | {"iterations": model.n_iter_}, model))
    auc, params, model = max(fits, key=lambda fit: fit[0])
    return model, params, auc


MODELS = {"logistic": fit_logistic, "boosting": fit_boosting}


def bootstrap_aucs(y: np.ndarray, p: np.ndarray, games: np.ndarray, resamples: int = 1000) -> np.ndarray:
    """Return the AUC of each bootstrap resample.

    Whole games are resampled, because windows from one game are not independent, and the seed is fixed so two
    models scored on the same split get the same resamples and their AUCs can be subtracted pairwise.
    """
    _, game = np.unique(games, return_inverse=True)
    count = game.max() + 1
    rng = np.random.default_rng(0)
    return np.array(
        [roc_auc_score(y, p, sample_weight=np.bincount(rng.integers(0, count, count), minlength=count)[game]) for _ in range(resamples)]
    )


def top_precision(y: np.ndarray, p: np.ndarray, share: float) -> float:
    """Return the blunder rate among the highest-scored share of windows.

    Windows tied at the cut count at their shared rate, so the answer does not depend on the order of the data.
    """
    count = int(np.ceil(share * len(y)))
    cut = np.sort(p)[-count]
    above, tied = p > cut, p == cut
    return float((y[above].sum() + (count - above.sum()) * y[tied].mean()) / count)


def evaluate(y: np.ndarray, p: np.ndarray, games: np.ndarray) -> dict:
    """Return every reported metric for one model's scores on one split."""
    base = float(y.mean())
    low, high = np.percentile(bootstrap_aucs(y, p, games), [2.5, 97.5])
    decile = top_precision(y, p, 0.1)
    return {
        "n": len(y),
        "auc": roc_auc_score(y, p),
        "auc_ci95": [low, high],
        "average_precision": average_precision_score(y, p),
        "base_rate": base,
        "brier": brier_score_loss(y, p),
        "top_decile_precision": decile,
        "top_decile_lift": decile / base,
        "top_5pct_precision": top_precision(y, p, 0.05),
    }


def _rounded(value):
    if isinstance(value, dict):
        return {key: _rounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rounded(item) for item in value]
    return round(float(value), 4) if isinstance(value, (float, np.floating)) else value


def main() -> None:
    """Train every baseline, write results/baselines.json and models/baseline-clocks.pkl, print the table."""
    windows = {split: load_windows(split) for split in ("train", "validation", "test")}
    y = {split: np.array([w.label for w in ws]) for split, ws in windows.items()}
    games = np.array([w.game_id for w in windows["test"]])
    buckets = np.array([speed_bucket(w) for w in windows["test"]])

    rows, test_scores, boosted = [], {}, {}
    for feature_set in FEATURE_SETS:
        x = {split: matrix(ws, feature_set) for split, ws in windows.items()}
        for model_name, fit in MODELS.items():
            model, params, validation_auc = fit(x["train"], y["train"], x["validation"], y["validation"])
            scores = model.predict_proba(x["test"])[:, 1]
            test_scores[feature_set, model_name] = scores
            if model_name == "boosting":
                boosted[feature_set] = model, model.predict_proba(x["validation"])[:, 1]
            rows.append(
                {"features": feature_set, "model": model_name, "n_features": x["train"].shape[1], "params": params,
                 "validation_auc": validation_auc, **evaluate(y["test"], scores, games)}
            )  # fmt: skip

    train_rate, test_rate = float(y["train"].mean()), float(y["test"].mean())
    reference = {  # a constant score ranks nothing, so its top-k precision is the base rate
        "features": "none", "model": "always the train base rate", "n": len(y["test"]), "auc": 0.5,
        "average_precision": test_rate, "base_rate": test_rate,
        "brier": brier_score_loss(y["test"], np.full(len(y["test"]), train_rate)),
        "top_decile_precision": test_rate, "top_decile_lift": 1.0, "top_5pct_precision": test_rate,
    }  # fmt: skip

    def gain(better: str, worse: str, rows=slice(None)) -> dict:
        labels, group = y["test"][rows], games[rows]
        a, b = test_scores[better, "boosting"][rows], test_scores[worse, "boosting"][rows]
        diff = bootstrap_aucs(labels, a, group) - bootstrap_aucs(labels, b, group)
        return {"auc_gain": roc_auc_score(labels, a) - roc_auc_score(labels, b), "auc_gain_ci95": list(np.percentile(diff, [2.5, 97.5]))}

    def bucket_report(rows: np.ndarray) -> dict:
        return {
            "n": int(rows.sum()),
            "base_rate": float(y["test"][rows].mean()),
            "auc": {fs: roc_auc_score(y["test"][rows], test_scores[fs, "boosting"][rows]) for fs in FEATURE_SETS},
            "clocks_over_priors": gain("clocks", "priors", rows),
        }

    model, validation_scores = boosted["clocks"]
    threshold = float(np.quantile(validation_scores, 0.9))
    called = test_scores["clocks", "boosting"] >= threshold
    report = {
        "dataset": DATASET,
        "label": "the player to move blunders within their next three moves",
        "selection": "trained on train, hyper-parameters chosen by validation AUC, every number below is on test",
        "bootstrap": "1,000 resamples of whole games, percentile 95% interval",
        "feature_sets": {
            "priors": "ratings, time control, ply and side to move: known without timing a single move",
            "clock-dynamics": "seconds left and seconds spent per move for both players; no rating or ply, but the seconds "
            "give the time control away, so this set is not free of priors",
            "clocks": "priors + clock-dynamics, all a live game offers; its gain over priors is what the clocks add",
            "clocks+engine": "clocks + the engine evaluation so far, which a live game does not have",
        },
        "splits": {split: {"n": len(labels), "base_rate": float(labels.mean())} for split, labels in y.items()},
        "reference": reference,
        "results": rows,
        "ablation_boosting": {
            "clocks_over_priors": gain("clocks", "priors"),
            "clocks_over_clock_dynamics": gain("clocks", "clock-dynamics"),
            "engine_over_clocks": gain("clocks+engine", "clocks"),
        },
        "speed_buckets_boosting": {bucket: bucket_report(buckets == bucket) for bucket in ("bullet", "blitz", "rapid", "classical")},
        # grouped after seeing the per-bucket numbers, so read it as a hint and not as a pre-planned test
        "post_hoc_all_but_bullet_boosting": bucket_report(buckets != "bullet"),
        "call_threshold": {
            "model": "clocks / boosting",
            "p": threshold,
            "meaning": "the 90th percentile of validation scores; the live watcher calls a blunder at or above it",
            "validation_precision": float(y["validation"][validation_scores >= threshold].mean()),
            "test_share_called": float(called.mean()),
            "test_precision": float(y["test"][called].mean()),
        },
    }
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(_rounded(report), indent=2) + "\n")
    BASELINE_MODEL.parent.mkdir(exist_ok=True)
    BASELINE_MODEL.write_bytes(pickle.dumps({"model": model, "threshold": threshold}))

    print(f"test n={len(y['test'])}, base rate {test_rate:.3f}")
    print(f"{'features':<15}{'model':<10}{'AUC':>6}  {'95% CI':<15}{'AP':>6}{'Brier':>7}{'top10%':>8}{'lift':>6}{'top5%':>7}")
    for row in rows:
        print(
            f"{row['features']:<15}{row['model']:<10}{row['auc']:>6.3f}  {row['auc_ci95'][0]:.3f}-{row['auc_ci95'][1]:.3f}    "
            f"{row['average_precision']:>6.3f}{row['brier']:>7.4f}{row['top_decile_precision']:>8.3f}"
            f"{row['top_decile_lift']:>6.2f}{row['top_5pct_precision']:>7.3f}"
        )
    print(f"{'none':<15}{'constant':<10}{0.5:>6.3f}  {'':<15}{test_rate:>6.3f}{reference['brier']:>7.4f}{test_rate:>8.3f}{1:>6.2f}{test_rate:>7.3f}")
    for name, item in report["ablation_boosting"].items():
        print(f"{name}: AUC {item['auc_gain']:+.3f} (95% CI {item['auc_gain_ci95'][0]:+.3f} to {item['auc_gain_ci95'][1]:+.3f})")
    for bucket, item in (report["speed_buckets_boosting"] | {"not bullet": report["post_hoc_all_but_bullet_boosting"]}).items():
        low, high = item["clocks_over_priors"]["auc_gain_ci95"]
        print(
            f"{bucket:<11}n={item['n']:<5} base {item['base_rate']:.3f}  " + "  ".join(f"{fs} {auc:.3f}" for fs, auc in item["auc"].items())
            + f"  clocks-priors {item['clocks_over_priors']['auc_gain']:+.3f} ({low:+.3f} to {high:+.3f})"
        )
    call = report["call_threshold"]
    print(f"call threshold P>={threshold:.3f}: calls {call['test_share_called']:.1%} of test windows, {call['test_precision']:.1%} of them blunder")
    print(f"wrote {RESULTS} and {BASELINE_MODEL} ({BASELINE_MODEL.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
