"""Load lichess/clock-blunder from the local TimeNet registry as plain arrays, cached per split."""

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np

DATASET = "lichess/clock-blunder@1.0.0"
CACHE = Path(__file__).resolve().parents[2] / ".cache"


@dataclass(frozen=True)
class Window:
    """One example: the plies before a player's move, and whether they blunder in their next three."""

    white_clock: np.ndarray  # seconds, one value per ply
    black_clock: np.ndarray
    evaluation: np.ndarray  # pawns from White's view, clipped to ±10
    time_control: str  # "base+increment" in seconds
    white_elo: int
    black_elo: int
    side_to_move: str  # "white" or "black"
    ply: int
    game_id: str
    prompt: str
    rationale: str
    label: int  # 1 = blunders within their next three moves


def _text(value) -> str:
    return str(getattr(value, "value", value))


def load_windows(split: str) -> list[Window]:
    """Return every window of a split ("train", "validation" or "test")."""
    cached = CACHE / f"{split}.pkl"
    if cached.exists():
        return pickle.loads(cached.read_bytes())
    import timenet  # slow import, only needed on a cache miss

    dataset = timenet.TimeNet().load(DATASET)
    windows = []
    for task in dataset.get_split(split):
        record = task.inputs[0]
        signals = {s.name: np.asarray(s.to_numpy(), dtype=np.float32) for s in record.signals}
        notes = {a.key: getattr(a.value, "value", a.value) for a in record.annotations}
        windows.append(
            Window(
                white_clock=signals["white_clock"],
                black_clock=signals["black_clock"],
                evaluation=signals["evaluation_pawns"],
                time_control=str(notes["time_control"]),
                white_elo=int(notes["white_elo"]),
                black_elo=int(notes["black_elo"]),
                side_to_move=str(notes["side_to_move"]),
                ply=int(notes["ply"]),
                game_id=str(record.metadata["lichess_game_id"]),
                prompt=_text(task.prompt),
                rationale=_text(task.rationale),
                label=int(_text(task.targets[0]) == "yes"),
            )
        )
    CACHE.mkdir(exist_ok=True)
    cached.write_bytes(pickle.dumps(windows))
    return windows
