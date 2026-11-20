"""Run the live watcher: uv run python -m gongoozler.live"""

import argparse
import json
import logging
import os

import uvicorn

from gongoozler.live.heuristic import DevHeuristicPredictor
from gongoozler.live.server import create_app
from gongoozler.live.verifier import DEPTH
from gongoozler.live.watcher import Watcher
from gongoozler.predict import ROOT, Predictor, load_predictor

FALLBACK_THRESHOLD = 0.40


def default_threshold() -> float:
    """The top-decile threshold the baselines report, if they have been run."""
    try:
        return float(json.loads((ROOT / "results" / "baselines.json").read_text())["call_threshold"]["p"])
    except (FileNotFoundError, KeyError):
        return FALLBACK_THRESHOLD


def pick_predictor(prefer: str | None) -> Predictor:
    if prefer != "dev":
        try:
            return load_predictor(prefer)
        except FileNotFoundError as error:
            logging.warning("%s; using the development heuristic, whose numbers mean nothing", error)
    return DevHeuristicPredictor()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model", choices=("tslm", "baseline", "dev"), help="default: the best trained model on disk")
    parser.add_argument("--threshold", type=float, help="call a blunder at P >= this (default: the baselines' top decile)")
    parser.add_argument("--games-per-speed", type=int, default=2, help="games followed in each of bullet, blitz, rapid")
    parser.add_argument("--depth", type=int, default=DEPTH, help="Stockfish depth per position")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    # scikit-learn's OpenMP threads otherwise spin between predictions: measured 154% of a core against 8%.
    os.environ.setdefault("OMP_WAIT_POLICY", "passive")
    threshold = default_threshold() if args.threshold is None else args.threshold
    watcher = Watcher(pick_predictor(args.model), threshold, per_speed=args.games_per_speed, depth=args.depth)
    logging.info("model: %s, calling at P >= %.3f, on http://%s:%d", watcher.predictor.name, threshold, args.host, args.port)
    uvicorn.run(create_app(watcher), host=args.host, port=args.port, log_level="warning", timeout_graceful_shutdown=2)


if __name__ == "__main__":
    main()
