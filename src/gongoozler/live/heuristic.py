"""A stand-in used only when no trained model is on disk, so the watcher can be developed and demoed end to end."""

from gongoozler.data import Window


class DevHeuristicPredictor:
    """Time left versus recent pace. Not trained on anything: its numbers say nothing about Gongoozler."""

    name = "development heuristic (no trained model on disk)"

    def p_blunder(self, window: Window) -> float:
        clock = window.white_clock if window.side_to_move == "white" else window.black_clock
        recent = clock[-7:]  # the mover's last three moves span six plies
        pace = max(float(recent[0] - recent[-1]) / 3, 0.5)
        return 0.1 + 0.5 / (1 + float(clock[-1]) / pace / 10)

    def explain(self, window: Window) -> str:
        clock = window.white_clock if window.side_to_move == "white" else window.black_clock
        return f"{window.side_to_move.capitalize()} has {clock[-1]:.0f} seconds left."
