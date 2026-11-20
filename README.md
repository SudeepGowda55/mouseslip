# Gardyloo

**It calls the blunder without seeing the board.**

"Gardyloo!" was the cry in old Edinburgh before something was thrown from a window: a warning, just before it lands.

Gardyloo is a time-series language model that follows a chess game the way a spectator with no view of the board would. It reads three lines, White's clock, Black's clock and the engine's evaluation bar, then says whether the player to move is about to blunder and why. A live watcher runs it on real Lichess games in progress, and Stockfish keeps score.

Built for the [TimeNet Online Hackathon](https://challenge.timenet.ai/) by team Matcrypt (Sudeep and Manvith).

> **Status, 20 November 2026.** The dataset, the baselines, the live watcher and the web page are done. The fine-tuned model is still training, so its test results are not in this README yet. Nothing below is a projected number.

## What is in here

| Part | What it does |
|---|---|
| **Dataset** `lichess/clock-blunder` | 60,358 windows from 15,205 real rated Lichess games, contributed to TimeNet in [OpenTSLM/TimeNet#834](https://github.com/OpenTSLM/TimeNet/pull/834) |
| **Model** `src/gardyloo/tslm.py` | Fine-tunes [OpenTSLM](https://github.com/OpenTSLM/OpenTSLM) (soft-prompt variant) on Gemma-3-270M with LoRA |
| **Baselines** `src/gardyloo/baselines.py` | Logistic regression and gradient boosting on hand-made features, with an ablation |
| **Live watcher** `src/gardyloo/live/` | Follows running Lichess games, calls blunders, lets Stockfish judge each call |
| **Web page** `web/` | The live games as three lines each, the calls and an honest scoreboard |

## The task

Each example is a window of at most 40 plies of one game, cut just before a player's move:

- **Input:** White's clock and Black's clock in seconds, and the Stockfish evaluation in pawns, one value per ply. No moves, no position.
- **Question:** will the player to move blunder within their next three moves?
- **Blunder:** Lichess's own rule. A move that drops the mover's winning chances by 0.3 or more on the -1 to +1 scale.

23% of the windows are a yes. The split is by game (48,055 train, 6,136 validation, 6,167 test), so no game is in two splits.

## How the model answers

The model first writes a short description of what it sees in the series, then answers:

```
White is low on time, is behind on the clock, at a steady pace. Board: White is about level,
evaluation steady. Blunders: White 0, Black 1. Answer: no
```

The probability of a blunder is read from the answer token, after the model's own description. The description is checked too: it can be compared word for word with the one computed from the same numbers.

An earlier version answered first and explained afterwards. It stalled at a validation AUC of 0.63, no better than knowing the ratings and the time control alone, so it was dropped.

## What the baselines say

Gradient boosting on hand-made features, on the 6,167 held-out test windows. AUC is the chance that a window with a blunder is ranked above one without; 0.5 is guessing.

| The model is given | AUC | 95% interval | Precision in its top 10% |
|---|---|---|---|
| Ratings, time control and move number only | 0.642 | 0.622 to 0.661 | 46% |
| Clock movement only | 0.640 | 0.620 to 0.659 | 43% |
| Both of the above | 0.650 | 0.631 to 0.668 | 42% |
| The above plus the evaluation bar | **0.785** | 0.771 to 0.800 | 63% |

**The honest finding:** the clocks alone add almost nothing. Knowing who is playing and how fast the game is already gives 0.642, and adding the full clock series moves it by 0.008 (interval -0.002 to +0.017). The evaluation bar carries the signal: it adds 0.135. The project started as "predict blunders from the clocks alone", and this table is why it no longer claims that.

## Run it

Needs Python 3.12, [uv](https://docs.astral.sh/uv/), Stockfish (`brew install stockfish`) and, for training, a Hugging Face token with the [Gemma licence](https://huggingface.co/google/gemma-3-270m) accepted.

```bash
git clone https://github.com/SudeepGowda55/gardyloo && cd gardyloo
uv sync

# the dataset (until the pull request is merged, build it from the fork)
git clone -b feature/lichess-clock-blunder https://github.com/SudeepGowda55/TimeNet
(cd TimeNet && uv sync --all-extras && uv pip install zstandard \
  && uv run timenet-build build lichess/clock-blunder --no-isolation)

uv run pytest                              # 44 tests, no model or network needed
uv run python -m gardyloo.baselines        # the table above
scripts/train_all.sh                       # both models; about 4.5 hours each on an M1
uv run python -m gardyloo.tslm explain     # a few test windows with the model's own words
uv run python -m gardyloo.live             # live watcher and web page on http://127.0.0.1:8765
uv run python scripts/demo.py              # the same calls in a terminal
```

`curl localhost:8765/api/state` returns everything the page shows, and `/api/results` the test results.

## How a live call is judged

1. The watcher follows a few running games per speed (bullet, blitz, rapid) through Lichess's public game streams.
2. Stockfish evaluates each new position. Those evaluations are the model's third line.
3. From move 7 on, the model scores every position. At or above the call threshold it issues a call.
4. Once the mover has played three more moves, Stockfish's evaluations decide: hit, miss, or void if the game ended first.
5. Every judged position is appended to a log, so the scoreboard adds up across restarts, separately per model and threshold.

Two things to know when reading the live numbers:

- Lichess holds its public streams back by a few seconds (longer in slower games). A call is made before the watcher has seen the following moves, not necessarily before they were played.
- The training evaluations are Lichess's own from 2017; the live ones come from a current Stockfish at depth 12. The scale is the same but the engines differ.

## Data handling

- Games come from the [Lichess open database](https://database.lichess.org/) (CC0) and from Lichess's public API. No account or key is used.
- Player names are not stored in the dataset, the live log, the API or the page. Ratings and titles are.

## Layout

| Path | What it is |
|---|---|
| `src/gardyloo/data.py` | Loads the dataset from the local TimeNet registry |
| `src/gardyloo/tslm.py` | Prompt building, training, evaluation, the model behind the predictor interface |
| `src/gardyloo/baselines.py` | Feature sets, baselines, ablation |
| `src/gardyloo/predict.py` | The one interface both models sit behind |
| `src/gardyloo/live/` | Feed, game state, Stockfish referee, watcher, HTTP server |
| `web/` | Static page served by the watcher |
| `scripts/` | Training script, watchdog, terminal demo |
| `results/` | Test results as JSON |
| `tests/` | 44 tests |

## Licence

MIT. The dataset keeps Lichess's CC0.
