"""Terminal run-of-show: Mouse Slip's calls and Stockfish's verdicts as they happen, with the running score.

Start the watcher first (uv run python -m mouseslip.live), then run: uv run python scripts/demo.py
"""

import argparse
import json
import time

import httpx


def share(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def score_line(board: dict) -> str:
    total = board["all"]
    lift = "n/a" if total["lift"] is None else f"{total['lift']:.2f}x"
    return (
        f"calls {total['calls']}, hits {total['hits']}, precision {share(total['precision'])} | base rate "
        f"{share(total['base_rate'])} over {total['resolved']} decision points | lift {lift} | void {total['void']}"
    )


def player(game: dict, side: str) -> str:
    title = game[side]["title"]
    return f"{side.capitalize()} ({title + ' ' if title else ''}{game[side]['rating']})"


def show(event: dict) -> None:
    game, call = event["game"], event.get("call")
    where = f"{game['speed']:<6} {game['time_control']:<6} {game['link']}"
    stamp = time.strftime("%H:%M:%S")
    if event["type"] == "call":
        print(f"{stamp} CALL  {where}  after ply {call['ply']}: {player(game, call['side'])} is about to blunder, P={call['p']:.2f}")
    elif event["type"] == "explain" and call["explain"]:
        print(f"{'':8} why   {call['explain']}")
    elif event["type"] == "resolution":
        if call["void"]:
            print(f"{stamp} VOID  {where}  call after ply {call['ply']}: the game ended before {call['side']}'s third move")
        else:
            verdict = "HIT " if call["blunder"] else "MISS"
            worst = f"worst of the three moves: {call['worst_move']} at ply {call['worst_ply']} lost {call['drop']:+.2f} winning chances"
            print(f"{stamp} {verdict}  {where}  call after ply {call['ply']} on {call['side']}: Stockfish says {worst}")
        print(f"{'':8} score {score_line(event['scoreboard'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    args = parser.parse_args()
    with httpx.Client(base_url=args.url, timeout=httpx.Timeout(10, read=60)) as http:
        while True:
            try:
                state = http.get("/api/state").json()
                print(f"Mouse Slip is watching {len(state['games'])} live Lichess games with the clocks only.")
                print(f"model: {state['model']}{'' if state['trained'] else '  ** NOT A TRAINED MODEL **'}")
                print(f"calls a blunder at P >= {state['threshold']:.3f}; Stockfish depth {state['verifier']['depth']} is the referee")
                print(f"so far: {score_line(state['scoreboard'])}\n")
                with http.stream("GET", "/api/events") as events:
                    for line in events.iter_lines():
                        if line.startswith("data: "):
                            show(json.loads(line[6:]))
            except httpx.TransportError as error:
                print(f"waiting for the watcher at {args.url} ({type(error).__name__})")
                time.sleep(3)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
