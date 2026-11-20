"""Ties it together: find games, score every new position, let Stockfish judge, keep the scoreboard."""

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime
import json
import logging
from pathlib import Path
import time

import httpx

from gardyloo.live import feed
from gardyloo.live.game import FIRST_DECISION_PLY, SPEEDS, Decision, Game, Scoreboard
from gardyloo.live.heuristic import DevHeuristicPredictor
from gardyloo.live.verifier import DEPTH, Verifier
from gardyloo.predict import ROOT, Predictor

LOG = ROOT / ".cache" / "live-log.jsonl"
REPLAY_SECONDS = 1.5  # positions this soon after connecting are the replay of moves already played
DISCOVER_SECONDS = 15
ARENA_SECONDS = 300  # how often the list of arenas is read again
LINGER_SECONDS = 30  # a finished game stays visible this long
MAX_FAILURES = 5

log = logging.getLogger("gardyloo.live")


class Watcher:
    """Follows a few games per speed and scores itself; all state lives in memory plus the append-only log."""

    def __init__(self, predictor: Predictor, threshold: float, *, per_speed: int = 2, log_path: Path = LOG, depth: int = DEPTH) -> None:
        self.predictor, self.threshold, self.per_speed, self.log_path = predictor, threshold, per_speed, log_path
        self.verifier = Verifier(depth=depth)
        self.games: dict[str, Game] = {}
        self.seen: set[str] = set()
        self.arenas: list[dict] = []
        self.arenas_at = 0.0
        self.board = Scoreboard()
        self.pending: dict[str, tuple[int, bool]] = {}  # game id -> (ply to score, arrived live); the latest ply wins
        self.wake = asyncio.Event()
        self.to_verify: asyncio.Queue[tuple[Game, int]] = asyncio.Queue()
        self.subscribers: set[asyncio.Queue] = set()
        self.tasks: set[asyncio.Task] = set()
        self.skipped = 0  # live positions the model was too slow to score before the next move came
        self.started = time.time()
        self.counting_since = datetime.now(UTC).isoformat(timespec="seconds")
        if log_path.exists():  # numbers add up across restarts, but never across models or thresholds
            for entry in map(json.loads, log_path.read_text().splitlines()):
                if (entry["model"], entry["threshold"]) == (predictor.name, threshold):
                    self.counting_since = min(self.counting_since, entry["t"])
                    self.board.add(entry["speed"], called=entry["called"], blunder=entry["blunder"])

    def state(self) -> dict:
        return {
            "model": self.predictor.name,
            "trained": not isinstance(self.predictor, DevHeuristicPredictor),
            "threshold": self.threshold,
            "scoreboard": self.board.snapshot(),
            "counting_since": self.counting_since,
            "games": [game.view() for game in self.games.values()],
            "verifier": {"engine": "Stockfish", "depth": self.verifier.depth, "queue": self.to_verify.qsize()},
            "skipped": self.skipped,
            "uptime_seconds": round(time.time() - self.started),
            "feed": "Lichess public game streams, which Lichess holds back by a few seconds (longer in slower games)",
        }

    def emit(self, kind: str, game: Game, **extra) -> None:
        event = {"type": kind, "t": time.time(), "game": game.view(), **extra}
        for queue in self.subscribers:
            if not queue.full():  # a reader that stopped reading misses events instead of holding memory
                queue.put_nowait(event)

    async def run(self) -> None:
        await self.verifier.start()
        try:
            async with feed.client() as self.http:
                await asyncio.gather(self.discover(), self.score_loop(), self.verify_loop())
        finally:
            for task in self.tasks:
                task.cancel()
            await self.verifier.close()

    async def discover(self) -> None:
        """Keep `per_speed` unfinished games per speed; drop finished ones once they have lingered."""
        while True:
            for game in list(self.games.values()):
                if game.finished and not game.open and time.time() - game.finished_at > LINGER_SECONDS:
                    del self.games[game.id]
            pause = DISCOVER_SECONDS
            for speed in SPEEDS:
                free = self.per_speed - sum(g.speed == speed and not g.finished for g in self.games.values())
                if free <= 0:
                    continue
                try:
                    async for game in self.candidates(speed):
                        if game.id in self.seen:
                            continue
                        self.seen.add(game.id)
                        self.games[game.id] = game
                        task = asyncio.create_task(self.follow(game))
                        self.tasks.add(task)
                        task.add_done_callback(self.tasks.discard)
                        free -= 1
                        if not free:
                            break
                except (httpx.HTTPError, ValueError, KeyError) as error:
                    log.warning("looking for %s games: %r", speed, error)
                    if getattr(getattr(error, "response", None), "status_code", 0) == 429:
                        pause = 60  # Lichess asks for a full minute after a 429
            await asyncio.sleep(pause)

    async def candidates(self, speed: str) -> AsyncIterator[Game]:
        """Yield games worth following: rating-capped arenas first, then the weakest of Lichess TV's best."""
        if time.time() - self.arenas_at > ARENA_SECONDS:
            self.arenas_at = time.time()
            self.arenas = await feed.capped_arenas(self.http)
        for arena in self.arenas:
            if arena["perf"]["key"] == speed:
                for game in await feed.arena_games(self.http, arena):
                    yield game
        for game in await feed.tv_games(self.http, speed):
            yield game

    async def follow(self, game: Game) -> None:
        """Read one game's stream to its end, reconnecting on errors; the replay on reconnect fills any gap."""
        failures = 0
        while not game.finished and failures < MAX_FAILURES:
            pause = min(2**failures, 30)
            try:
                connected = None
                async for message in feed.stream_game(self.http, game.id):
                    now = time.monotonic()
                    connected = connected or now
                    if game.feed(message):
                        failures = 0
                        self.on_position(game, live=now - connected > REPLAY_SECONDS)
            except KeyError:
                game.status = "lost"  # a player without a rating
            except (httpx.HTTPError, ValueError) as error:
                log.warning("stream %s: %r", game.id, error)
                if getattr(getattr(error, "response", None), "status_code", 0) == 429:
                    pause = 60
            if not game.finished:
                failures += 1
                await asyncio.sleep(pause)
        if not game.finished:
            game.status = "lost"
        game.finished_at = time.time()
        self.settle(game)
        self.emit("game", game)

    def on_position(self, game: Game, *, live: bool) -> None:
        if live:
            self.emit("move", game)
        if game.ply >= FIRST_DECISION_PLY:
            if game.awaiting and game.awaiting[1]:
                self.skipped += 1
            game.awaiting = (game.ply, live)
        self.queue_positions(game)

    async def score_loop(self) -> None:
        """Score one position at a time, off the event loop; a decision counts only if it beats the next move."""
        while True:
            while not self.pending:
                self.wake.clear()
                await self.wake.wait()
            game_id = next(iter(self.pending))
            ply, live = self.pending.pop(game_id)
            game = self.games.get(game_id)
            if game is None:
                continue
            try:
                window = game.window(ply)
                p = game.scores[ply] = float(await asyncio.to_thread(self.predictor.p_blunder, window))
                if not live or game.finished:
                    self.emit("score", game)
                elif game.ply != ply:
                    self.skipped += 1
                    self.emit("score", game)
                else:
                    called = p >= self.threshold
                    game.open[ply] = decision = Decision(ply, window.side_to_move, p, called)
                    if called:  # the call goes out first: a language model may take seconds to word its reason
                        self.emit("call", game, call=decision.view())
                        decision.explain = await asyncio.to_thread(self.predictor.explain, window)
                        self.emit("explain", game, call=decision.view())
                    else:
                        self.emit("score", game)
            except Exception:
                log.exception("scoring %s at ply %d", game_id, ply)

    def queue_positions(self, game: Game) -> None:
        """Send the verifier every position, each once: the model reads the evaluations and the referee uses them."""
        while game.verify_next <= game.ply:
            self.to_verify.put_nowait((game, game.verify_next))
            game.verify_next += 1

    async def verify_loop(self) -> None:
        while True:
            game, position = await self.to_verify.get()
            game.pawns[position], game.chances[position] = await self.verifier.evaluate(game.fens[position])
            self.settle(game)
            # A position is scored only once it is evaluated; positions come in order, so the window is complete.
            if game.awaiting and game.awaiting[0] == position:
                if self.pending.get(game.id, (0, False))[1]:
                    self.skipped += 1
                self.pending[game.id], game.awaiting = game.awaiting, None
                self.wake.set()

    def settle(self, game: Game) -> None:
        for decision in game.settle():
            self.board.add(game.speed, called=decision.called, blunder=decision.blunder)
            entry = {
                "t": datetime.now(UTC).isoformat(timespec="seconds"),
                "game": game.id,
                "speed": game.speed,
                "time_control": game.time_control,
                "white_elo": game.white["rating"],
                "black_elo": game.black["rating"],
                "ply": decision.ply,
                "side": decision.side,
                "p": round(decision.p, 4),
                "called": decision.called,
                "blunder": decision.blunder,  # null = void: the game ended before the mover's third move
                "drop": decision.view()["drop"],
                "model": self.predictor.name,
                "threshold": self.threshold,
            }
            self.log_path.parent.mkdir(exist_ok=True)
            with self.log_path.open("a") as file:
                file.write(json.dumps(entry) + "\n")
            if decision.called:
                game.resolved_calls.appendleft(decision)
                self.emit("resolution", game, call=decision.view(), scoreboard=self.board.snapshot())
