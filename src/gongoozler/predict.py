"""The one interface every model sits behind, so the live watcher does not care which one is loaded."""

from pathlib import Path
from typing import Protocol

import numpy as np

from gongoozler.data import Window

ROOT = Path(__file__).resolve().parents[2]
TSLM_CHECKPOINT = ROOT / "checkpoints" / "tslm-clocks.pt"
BASELINE_MODEL = ROOT / "models" / "baseline-clocks.pkl"


class Predictor(Protocol):
    """Scores a window using the two clocks only: live games have no engine evaluation yet."""

    name: str

    def p_blunder(self, window: Window) -> float:
        """Return the probability that the player to move blunders within their next three moves."""
        ...

    def explain(self, window: Window) -> str:
        """Return one or two sentences on what in the clocks led to the score."""
        ...


def make_window(  # noqa: PLR0913
    clocks: list[float], time_control: str, white_elo: int, black_elo: int, *, game_id: str = "", window_plies: int = 40
) -> Window:
    """Build the window a model sees from a game so far.

    Args:
        clocks: The mover's clock in seconds after each ply played so far (ply 0 is White's first move).
            Needs at least two plies.
        time_control: "base+increment" in seconds.
        white_elo: White's rating.
        black_elo: Black's rating.
        game_id: The Lichess game id, kept for display.
        window_plies: The most plies a window holds, as in the dataset.

    Returns:
        The window for the player to move next, with no evaluation, prompt, rationale or label.
    """
    ply = len(clocks)
    white, black = np.empty(ply, np.float32), np.empty(ply, np.float32)
    last_white, last_black = clocks[0], clocks[1]  # a berserked player starts below the base time
    for index, value in enumerate(clocks):
        if index % 2 == 0:
            last_white = value
        else:
            last_black = value
        white[index], black[index] = last_white, last_black
    start = max(0, ply - window_plies)
    return Window(
        white_clock=white[start:],
        black_clock=black[start:],
        evaluation=np.zeros(ply - start, np.float32),
        time_control=time_control,
        white_elo=white_elo,
        black_elo=black_elo,
        side_to_move="white" if ply % 2 == 0 else "black",
        ply=ply,
        game_id=game_id,
        prompt="",
        rationale="",
        label=-1,
    )


def load_predictor(prefer: str | None = None) -> Predictor:
    """Load the best model available: the fine-tuned TSLM, else the gradient-boosting baseline.

    Args:
        prefer: "tslm" or "baseline" to force one.

    Raises:
        FileNotFoundError: If no trained model is on disk.
    """
    if prefer != "baseline" and TSLM_CHECKPOINT.exists():
        from gongoozler.tslm import TslmPredictor  # noqa: PLC0415 (torch is slow to import)

        return TslmPredictor(TSLM_CHECKPOINT)
    if prefer != "tslm" and BASELINE_MODEL.exists():
        from gongoozler.baselines import BaselinePredictor  # noqa: PLC0415

        return BaselinePredictor(BASELINE_MODEL)
    raise FileNotFoundError("no trained model found; run the training or the baselines first")
