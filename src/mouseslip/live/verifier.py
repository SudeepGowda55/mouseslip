"""Stockfish as the evaluation bar and the referee: every position in pawns and in winning chances, White's view."""

import chess
import chess.engine

from mouseslip.live.game import winning_chances

STOCKFISH = "/opt/homebrew/bin/stockfish"
# Depth 12 on one thread took about 70 ms a position on an M1 and stayed within 0.05 winning chances of depth 18
# on a sample game, far inside the 0.3 a blunder needs; six games produce about three positions a second.
DEPTH = 12


def terminal_chances(board: chess.Board) -> float | None:
    """Return the result of a finished position, or None if the game goes on."""
    if board.is_checkmate():
        return -1.0 if board.turn == chess.WHITE else 1.0
    return 0.0 if board.is_stalemate() or board.is_insufficient_material() else None


def score_chances(score: chess.engine.Score) -> float:
    """Turn an engine score from White's view into winning chances; a forced mate is ±1."""
    if score.is_mate():
        return 1.0 if score > chess.engine.Cp(0) else -1.0
    return winning_chances(score.score())


def score_pawns(score: chess.engine.Score) -> float:
    """Turn an engine score into what the dataset stores: pawns clipped to ±10, a forced mate at the limit."""
    if score.is_mate():
        return 10.0 if score > chess.engine.Cp(0) else -10.0
    return max(-10.0, min(10.0, score.score() / 100))


class Verifier:
    """One Stockfish process, asked one position at a time."""

    def __init__(self, path: str = STOCKFISH, depth: int = DEPTH) -> None:
        self.path, self.depth = path, depth
        self.engine: chess.engine.UciProtocol | None = None

    async def start(self) -> None:
        _, self.engine = await chess.engine.popen_uci(self.path)
        await self.engine.configure({"Threads": 1, "Hash": 32})

    async def evaluate(self, fen: str) -> tuple[float, float]:
        """Return a position's evaluation in pawns (what the model reads) and in winning chances (what judges it)."""
        board = chess.Board(fen)
        if (result := terminal_chances(board)) is not None:
            return 10.0 * result, result
        try:
            info = await self.engine.analyse(board, chess.engine.Limit(depth=self.depth))
        except chess.engine.EngineTerminatedError:  # one fresh process, then give up loudly
            await self.start()
            info = await self.engine.analyse(board, chess.engine.Limit(depth=self.depth))
        return score_pawns(info["score"].white()), score_chances(info["score"].white())

    async def close(self) -> None:
        if self.engine:
            await self.engine.quit()
