"""Record a running watcher (state, test results and every event) so the page can replay it without a server."""

import argparse
import json
import time
from pathlib import Path

import httpx

OUT = Path(__file__).resolve().parents[1] / "web" / "replay" / "recording.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--calls", type=int, default=3, help="stop once this many calls have been resolved")
    parser.add_argument("--minutes", type=float, default=90, help="stop after this long regardless")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    with httpx.Client(base_url=args.url, timeout=None) as http:
        recording = {
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "state": http.get("/api/state").raise_for_status().json(),
            "results": http.get("/api/results").raise_for_status().json(),
            "events": [],
        }
        start, resolved = time.time(), 0
        args.out.parent.mkdir(parents=True, exist_ok=True)
        try:
            with http.stream("GET", "/api/events") as stream:
                for line in stream.iter_lines():
                    if not line.startswith("data: "):
                        if time.time() - start > args.minutes * 60:
                            break
                        continue
                    event = json.loads(line[6:])
                    recording["events"].append(event)
                    if event["type"] == "resolution" and event.get("call", {}).get("blunder") is not None:
                        resolved += 1
                        print(f"{len(recording['events'])} events, {resolved} calls resolved", flush=True)
                    if resolved >= args.calls or time.time() - start > args.minutes * 60:
                        break
        finally:
            args.out.write_text(json.dumps(recording, separators=(",", ":")))
            print(f"wrote {args.out} ({len(recording['events'])} events)")


if __name__ == "__main__":
    main()
