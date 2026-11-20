"""The Lichess side: which games are on, and each game's positions and clocks as they come in."""

from collections.abc import AsyncIterator
import json

import httpx

from gardyloo.live.game import Game

LICHESS = "https://lichess.org"


def client() -> httpx.AsyncClient:
    # A long think in a rapid game sends nothing for minutes and Lichess sends no keep-alive lines.
    return httpx.AsyncClient(base_url=LICHESS, timeout=httpx.Timeout(15, read=300))


async def capped_arenas(http: httpx.AsyncClient) -> list[dict]:
    """Return the running rated standard arenas with a rating cap.

    Their players are rated like the games the models were trained on (median 1600); Lichess TV only shows the
    best games on the site, 2100 to 3000.
    """
    response = await http.get("/api/tournament")
    response.raise_for_status()
    return [a for a in response.json()["started"] if a.get("rated") and a["variant"]["key"] == "standard" and "maxRating" in a]


async def arena_games(http: httpx.AsyncClient, arena: dict) -> list[Game]:
    """Return the games an arena lists as going on right now (its top boards, six at most)."""
    response = await http.get(f"/api/tournament/{arena['id']}")
    response.raise_for_status()
    time_control = f"{arena['clock']['limit']}+{arena['clock']['increment']}"
    return [Game(duel["id"], arena["perf"]["key"], time_control, arena["fullName"]) for duel in response.json().get("duels", [])]


async def tv_games(http: httpx.AsyncClient, speed: str, count: int = 30) -> list[Game]:
    """Return the rated standard games going on among a TV channel's best, the lowest rated first."""
    response = await http.get(f"/api/tv/{speed}", params={"nb": count, "moves": "false"}, headers={"Accept": "application/x-ndjson"})
    response.raise_for_status()
    entries = [
        entry
        for entry in map(json.loads, response.text.splitlines())
        if (entry.get("status"), entry.get("variant"), entry.get("speed")) == ("started", "standard", speed)
        and entry.get("rated")
        and "clock" in entry
    ]
    entries.sort(key=lambda entry: sum(entry["players"][side].get("rating", 9999) for side in ("white", "black")))
    return [Game(e["id"], speed, f"{e['clock']['initial']}+{e['clock']['increment']}", "Lichess TV") for e in entries]


async def stream_game(http: httpx.AsyncClient, game_id: str) -> AsyncIterator[dict]:
    """Yield the lines of one game's stream: a description, every position from move one, then a closing description.

    Lichess holds this stream back as an anti-cheating measure: measured about 2 s behind the board in bullet,
    15 s in blitz and 40 s in rapid.
    """
    async with http.stream("GET", f"/api/stream/game/{game_id}") as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if line:
                yield json.loads(line)
