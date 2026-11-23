"""mouseslip: the command line. Each command runs the module that does the work."""

import argparse
import hashlib
import runpy
import sys

MODEL_URL = "https://github.com/SudeepGowda55/mouseslip/releases/download/model-v1/tslm-clocks+engine.pt"
MODEL_SHA256 = "6da707c5b19672bb529d024c4d2a2a985d1f12fe6d0357fc90e03e4e4b841138"

COMMANDS = {
    "live": "live watcher and web page on http://127.0.0.1:8765",
    "demo": "the watcher's calls and Stockfish's verdicts in this terminal (start `live` first)",
    "explain": "a few test windows with the model's own words",
    "baselines": "train and score the baselines, print the comparison table",
    "fetch-model": "download the trained TSLM (24 MB) from the GitHub release",
}


def fetch_model() -> None:
    import httpx  # noqa: PLC0415

    from mouseslip.predict import TSLM_CHECKPOINT  # noqa: PLC0415

    if TSLM_CHECKPOINT.exists():
        print(f"already there: {TSLM_CHECKPOINT}")
        return
    TSLM_CHECKPOINT.parent.mkdir(exist_ok=True)
    partial = TSLM_CHECKPOINT.with_suffix(".part")
    digest = hashlib.sha256()
    with httpx.stream("GET", MODEL_URL, follow_redirects=True, timeout=60) as response, partial.open("wb") as file:
        response.raise_for_status()
        for chunk in response.iter_bytes():
            digest.update(chunk)
            file.write(chunk)
    if digest.hexdigest() != MODEL_SHA256:
        partial.unlink()
        sys.exit("the download did not match the published checksum; nothing was saved")
    partial.rename(TSLM_CHECKPOINT)
    print(f"saved {TSLM_CHECKPOINT}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="mouseslip", description="Calls chess blunders from two clocks and the evaluation bar.")
    parser.add_argument("command", choices=COMMANDS, help="; ".join(f"{k}: {v}" for k, v in COMMANDS.items()))
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:  # noqa: PLR2004
        parser.parse_args()  # prints help or the error
    args, rest = parser.parse_args(sys.argv[1:2]), sys.argv[2:]  # the rest belongs to the command's own parser
    sys.argv = [f"mouseslip {args.command}", *rest]

    if args.command == "fetch-model":
        fetch_model()
    elif args.command == "live":
        from mouseslip.predict import TSLM_CHECKPOINT  # noqa: PLC0415

        if not TSLM_CHECKPOINT.exists() and "--model" not in rest:
            print("The trained TSLM is not on disk, so the baseline will make the calls. Run `mouseslip fetch-model` for the TSLM.")
        from mouseslip.live.__main__ import main as live  # noqa: PLC0415

        live()
    elif args.command == "demo":
        from mouseslip.predict import ROOT  # noqa: PLC0415

        runpy.run_path(str(ROOT / "scripts" / "demo.py"), run_name="__main__")
    elif args.command == "explain":
        from mouseslip.tslm import main as tslm  # noqa: PLC0415

        sys.argv = ["mouseslip", "explain", *rest]
        tslm()
    elif args.command == "baselines":
        from mouseslip.baselines import main as baselines  # noqa: PLC0415

        baselines()
