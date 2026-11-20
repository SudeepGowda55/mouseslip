"""Game tracking, the blunder rule and the scoreboard: pure logic, no network and no engine."""

from collections import defaultdict, deque
from dataclasses import dataclass, field
import math
import time

from gongoozler.data import Window
from gongoozler.predict import make_window

FIRST_DECISION_PLY = 12  # the dataset has no earlier windows
WINDOW_PLIES = 40  # the most plies a window holds, as in the dataset
HORIZON = 5  # the mover's next three moves are plies n+1, n+3 and n+5
BLUNDER_DROP = 0.3
SPEEDS = ("bullet", "blitz", "rapid")


def winning_chances(centipawns: float) -> float:
    """Lichess's winning chances on its -1..+1 scale; mates are ±1 and never come through here."""
    return 2 / (1 + math.exp(-0.00368208 * max(-10000.0, min(10000.0, centipawns)))) - 1


def chances_lost(before: float, after: float, *, white_moved: bool) -> float:
    """Winning chances the mover gave away with one move; both values are from White's view."""
    return before - after if white_moved else after - before


def _player(player: dict) -> dict:
    return {"rating": int(player["rating"]), "title": player.get("user", {}).get("title")}  # never the username


def _nobody() -> dict:
    return {"rating": None, "title": None}


@dataclass
class Decision:
    """One scored position: will the player to move blunder within their next three moves?"""

    ply: int  # plies played when it was scored
    side: str
    p: float
    called: bool
    explain: str = ""
    t: float = field(default_factory=time.time)  # when it was scored, so a page loaded later can show the hour
    blunder: bool | None = None  # None while open, and for good when void
    void: bool = False
    drop: float | None = None  # the most winning chances lost in one of the three moves
    worst_ply: int | None = None
    worst_move: str = ""

    def view(self) -> dict:
        return {
            "ply": self.ply,
            "side": self.side,
            "p": round(self.p, 4),
            "t": self.t,
            "explain": self.explain,
            "blunder": self.blunder,
            "void": self.void,
            "drop": None if self.drop is None else round(self.drop, 3),
            "worst_ply": self.worst_ply,
            "worst_move": self.worst_move,
        }


@dataclass
class Game:
    """A followed game; every list is indexed by plies played, so index 0 is the starting position."""

    id: str
    speed: str
    time_control: str
    source: str = ""  # where it was found: "Lichess TV" or an arena's name
    white: dict = field(default_factory=_nobody)  # rating and title, set by the stream's opening line
    black: dict = field(default_factory=_nobody)
    status: str = "started"
    fens: list[str] = field(default_factory=list)
    moves: list[str] = field(default_factory=list)
    white_clock: list[float] = field(default_factory=list)
    black_clock: list[float] = field(default_factory=list)
    chances: dict[int, float] = field(default_factory=dict)  # Stockfish's verdict per position, White's view
    pawns: dict[int, float] = field(default_factory=dict)  # the same verdict in pawns, as the model reads it
    awaiting: tuple[int, bool] | None = None  # the latest position still to be evaluated before it is scored
    scores: dict[int, float] = field(default_factory=dict)  # P(blunder) per scored position
    open: dict[int, Decision] = field(default_factory=dict)
    resolved_calls: deque = field(default_factory=lambda: deque(maxlen=12))
    verify_next: int = 0  # first position not yet sent to the verifier
    finished_at: float = 0.0

    @property
    def ply(self) -> int:
        return len(self.fens) - 1

    @property
    def finished(self) -> bool:
        return self.status != "started"

    def feed(self, message: dict) -> bool:
        """Take one line of /api/stream/game/<id>; return True if it added a position.

        The stream replays the game from move one on every connection, so known positions are skipped.

        Raises:
            KeyError: If a player has no rating.
            ValueError: If a position is missing in between.
        """
        if "id" in message:  # the opening description, or the closing one, which alone carries a status
            if "status" in message:
                self.status = message["status"]["name"]
            else:
                self.white, self.black = (_player(message["players"][side]) for side in ("white", "black"))
            return False
        _, turn, *_, fullmove = message["fen"].split()
        ply = (int(fullmove) - 1) * 2 + (turn == "b")
        if ply != len(self.fens):
            if ply > len(self.fens):
                raise ValueError(f"{self.id}: position {ply} arrived after {len(self.fens) - 1}")
            return False
        self.fens.append(message["fen"])
        self.moves.append(message.get("lm", ""))
        self.white_clock.append(float(message["wc"]))
        self.black_clock.append(float(message["bc"]))
        return True

    def window(self, ply: int) -> Window:
        """Return what the model sees once `ply` plies are played: the mover's clock and the evaluation after each.

        Raises:
            KeyError: If a position of the window has no evaluation yet.
        """
        clocks = [(self.white_clock if i % 2 else self.black_clock)[i] for i in range(1, ply + 1)]
        first = max(1, ply - WINDOW_PLIES + 1)  # only the window's own positions must be evaluated
        evaluations = [self.pawns[i] if i >= first else 0.0 for i in range(1, ply + 1)]
        return make_window(
            clocks, self.time_control, self.white["rating"], self.black["rating"], evaluations=evaluations, game_id=self.id
        )

    def settle(self) -> list[Decision]:
        """Close every decision that can be closed and return them.

        A decision resolves once the mover's three moves are played and evaluated. If the game ended first it is
        void: neither a hit nor a miss, even if a blunder was already on the board.
        """
        closed = []
        for ply, decision in list(self.open.items()):
            if ply + HORIZON > self.ply:
                if not self.finished:
                    continue
                decision.void = True
            elif all(i in self.chances for i in range(ply, ply + HORIZON + 1)):
                drops = {
                    i + 1: chances_lost(self.chances[i], self.chances[i + 1], white_moved=decision.side == "white")
                    for i in range(ply, ply + HORIZON, 2)
                }
                decision.worst_ply = max(drops, key=drops.get)
                decision.worst_move = self.moves[decision.worst_ply]
                decision.drop = drops[decision.worst_ply]
                decision.blunder = decision.drop >= BLUNDER_DROP
            else:
                continue
            del self.open[ply]
            closed.append(decision)
        return closed

    def view(self, tail: int = 80) -> dict:
        """Return the game as the API shows it."""
        start = max(0, len(self.fens) - tail)
        return {
            "id": self.id,
            "link": f"https://lichess.org/{self.id}",
            "speed": self.speed,
            "time_control": self.time_control,
            "source": self.source,
            "white": self.white,
            "black": self.black,
            "status": self.status,
            "ply": self.ply,
            "side_to_move": "white" if self.ply % 2 == 0 else "black",
            "fen": self.fens[-1] if self.fens else None,
            "last_move": self.moves[-1] if self.moves else "",
            "clocks": {"white": self.white_clock[-1], "black": self.black_clock[-1]} if self.fens else None,
            "series": {
                "from_ply": start,
                "white": self.white_clock[start:],
                "black": self.black_clock[start:],
                "evaluation": [self.pawns.get(i) for i in range(start, len(self.fens))],  # pawns, White's view
                "p": [[ply, round(p, 4)] for ply, p in self.scores.items() if ply >= start],
            },
            "p": round(self.scores[self.ply], 4) if self.ply in self.scores else None,
            "open_decisions": len(self.open),
            "open_calls": [d.view() for d in self.open.values() if d.called],
            "resolved_calls": [d.view() for d in self.resolved_calls],
        }


class Scoreboard:
    """Counts per speed bucket and overall; a void decision is counted apart and never as a hit or a miss."""

    def __init__(self) -> None:
        self.counts = defaultdict(lambda: {"resolved": 0, "blunders": 0, "calls": 0, "hits": 0, "void": 0})

    def add(self, speed: str, *, called: bool, blunder: bool | None) -> None:
        for key in ("all", speed):
            count = self.counts[key]
            if blunder is None:
                count["void"] += 1
                continue
            count["resolved"] += 1
            count["blunders"] += blunder
            count["calls"] += called
            count["hits"] += called and blunder

    def snapshot(self) -> dict:
        """Return the counts with base rate, precision and lift (None until they can be computed)."""
        board = {}
        for key in ("all", *SPEEDS, *self.counts):
            count = self.counts[key]
            base_rate = count["blunders"] / count["resolved"] if count["resolved"] else None
            precision = count["hits"] / count["calls"] if count["calls"] else None
            board[key] = count | {
                "base_rate": base_rate,
                "precision": precision,
                "lift": precision / base_rate if precision is not None and base_rate else None,
            }
        return board
