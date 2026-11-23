# Mouse Slip

**It calls the blunder without seeing the board.**

A mouse slip is chess slang for a blunder made by a slip of the hand: you meant one move and played another.

Mouse Slip is a time-series language model that follows a chess game the way a spectator with no view of the board would. It reads three lines, White's clock, Black's clock and the engine's evaluation bar, then says whether the player to move is about to blunder and why. A live watcher runs it on real Lichess games in progress, and Stockfish keeps score.

**Try it in your browser:** https://mouseslip.vercel.app (replays a real live session; the Run it page has the install steps)

**Demo, 2:36, narrated:** [docs/mouseslip-demo.mp4](docs/mouseslip-demo.mp4), recorded from the live page while the watcher followed real games.

Built for the [TimeNet Online Hackathon](https://challenge.timenet.ai/) by team Matcrypt (Sudeep and Manvith).

> **Result in one line.** Fine-tuned on 48,055 real game windows, the model reaches a test AUC of 0.741 writing its own description first, and 0.763 when handed the computed one. That is well above what the clocks and ratings alone give (0.650), and below a gradient-boosting baseline on the same three lines (0.785). Every number below is from a real run on held-out games.

## What is in here

| Part | What it does |
|---|---|
| **Dataset** `lichess/clock-blunder` | 60,358 windows from 15,205 real rated Lichess games, contributed to TimeNet in [OpenTSLM/TimeNet#834](https://github.com/OpenTSLM/TimeNet/pull/834) |
| **Model** `src/mouseslip/tslm.py` | Fine-tunes [OpenTSLM](https://github.com/OpenTSLM/OpenTSLM) (soft-prompt variant) on Gemma-3-270M with LoRA |
| **Baselines** `src/mouseslip/baselines.py` | Logistic regression and gradient boosting on hand-made features, with an ablation |
| **Live watcher** `src/mouseslip/live/` | Follows running Lichess games, calls blunders, lets Stockfish judge each call |
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

The description is a fixed function of the numbers (it never looks at the answer), so it can also be computed and handed to the model instead of generated. The results below report both.

## What the baselines say

Gradient boosting on hand-made features, on the 6,167 held-out test windows. AUC is the chance that a window with a blunder is ranked above one without; 0.5 is guessing.

| The model is given | AUC | 95% interval | Precision in its top 10% |
|---|---|---|---|
| Ratings, time control and move number only | 0.642 | 0.622 to 0.661 | 46% |
| Clock movement only | 0.640 | 0.620 to 0.659 | 43% |
| Both of the above | 0.650 | 0.631 to 0.668 | 42% |
| The above plus the evaluation bar | **0.785** | 0.771 to 0.800 | 63% |

**The honest finding:** the clocks alone add almost nothing. Knowing who is playing and how fast the game is already gives 0.642, and adding the full clock series moves it by 0.008 (interval -0.002 to +0.017). The evaluation bar carries the signal: it adds 0.135. The project started as "predict blunders from the clocks alone", and this table is why it no longer claims that.

## What the fine-tuned model scores

OpenTSLM-SP on Gemma-3-270M with LoRA, one epoch over the training split on an M1 laptop (about 4.5 hours). The checkpoint with the best validation AUC (step 7,000 of 12,013) was scored once on the same 6,167 test windows as the baselines.

| Model | AUC | 95% interval | Precision in its top 10% |
|---|---|---|---|
| Gradient boosting on clocks + ratings (from the table above) | 0.650 | 0.631 to 0.668 | 42% |
| **Mouse Slip, answering after its own description** | **0.741** | 0.727 to 0.755 | 60% |
| **Mouse Slip, answering after the computed description** | **0.763** | 0.749 to 0.776 | 61% |
| Gradient boosting on the same three lines + ratings (from the table above) | 0.785 | 0.771 to 0.800 | 63% |
| The same model before the padding fix (below) | 0.623 | 0.604 to 0.639 | 43% |

In the top tenth of its scores, 60% of the windows really are followed by a blunder, against a base rate of 23%: 2.6 times better than picking at random.

**It does not beat gradient boosting.** A 270M-parameter language model reading three raw series lands 0.02 to 0.04 AUC below a tree model given hand-made features of the same series. What it adds is the sentence: every call says why, in words a spectator can check.

**How good are its own descriptions?** On 400 test windows it wrote exactly the computed description 33% of the time (37% over the full test split). By field:

| Part of the description | Correct |
|---|---|
| Time left, clock gap, pace | 97% to 99% |
| Who is better on the board | 99% |
| How the evaluation is moving (steady, shifting, swinging) | 66% |
| Blunder counts so far for each side | 57% and 59% |

It reads levels well and counts events poorly. That gap is the whole difference between the two AUC rows: when the counts are handed to it, it uses them.

By speed (answering after its own description): bullet 0.779 (2,319 windows), blitz 0.676 (1,960), rapid 0.676 (1,624), classical 0.725 (264).

## The bug that cost the first run

The first full training run scored 0.623 on test, below the 0.642 that ratings and time control give on their own. A per-field check of the descriptions showed why: the model got even the "who is better" field wrong, although that field is written in the prompt.

Gemma's tokenizer pads on the **left**. OpenTSLM's `pad_and_apply_batch` tokenizes the text segments of a batch with padding and then keeps the **first** n positions of each one, which assumes right padding. Every segment shorter than the longest in its batch, which is every series label and the question, was replaced by padding. The model was trained on the first sentence of the prompt and three unlabelled series.

The fix is one line in `load_model` (`model.tokenizer.padding_side = "right"`) plus an assert that fails if it ever regresses. After it, the same recipe went from 0.623 to 0.741. The first run's results are kept in `results/tslm-clocks+engine-before-padding-fix.json`.

## Live

The watcher ran the fine-tuned model on real Lichess games in progress on 21 to 23 November 2026, with Stockfish judging each call three moves later. It calls a blunder when its score reaches the top tenth of its test scores (0.293). Snapshot at 03:00 IST on 23 November:

| | Count |
|---|---|
| Positions judged | 1,891 |
| Followed by a blunder within three moves | 400 (21.2%) |
| Calls made | 23 (11 rapid, 8 blitz, 4 bullet) |
| Calls that were right | **11 of 23 (48%)** |

When it called a blunder, it was right 2.3 times as often as the base rate. With only 23 calls the 95% interval for that hit rate is wide, 29% to 67%, but even its low end is above the 21.2% base rate. The web page shows the running count, separately for each model and threshold, and it carries across restarts.

## Run it

Needs Python 3.12, [uv](https://docs.astral.sh/uv/), Stockfish (`brew install stockfish`) and, for training, a Hugging Face token with the [Gemma licence](https://huggingface.co/google/gemma-3-270m) accepted.

```bash
git clone https://github.com/SudeepGowda55/mouseslip && cd mouseslip
uv sync

# the dataset (until the pull request is merged, build it from the fork)
git clone -b feature/lichess-clock-blunder https://github.com/SudeepGowda55/TimeNet
(cd TimeNet && uv sync --all-extras && uv pip install zstandard \
  && uv run timenet-build build lichess/clock-blunder --no-isolation)

uv run mouseslip fetch-model    # the trained TSLM from the release (24 MB, checksum checked)
uv run mouseslip live           # live watcher and web page on http://127.0.0.1:8765
uv run mouseslip demo           # the same calls in a terminal, while `live` runs
uv run mouseslip explain        # a few test windows with the model's own words
uv run mouseslip baselines      # the table above
uv run pytest                   # 46 tests, no model or network needed
scripts/train_all.sh            # retrain instead of downloading: about 4.5 hours per model on an M1
```

The TSLM loads Gemma-3-270M from Hugging Face, so accept its licence and run `uv run hf auth login` once. Without the trained TSLM, `live` makes its calls with the gradient-boosting baseline (shipped in `models/`).

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
| `src/mouseslip/data.py` | Loads the dataset from the local TimeNet registry |
| `src/mouseslip/tslm.py` | Prompt building, training, evaluation, the model behind the predictor interface |
| `src/mouseslip/baselines.py` | Feature sets, baselines, ablation |
| `src/mouseslip/predict.py` | The one interface both models sit behind |
| `src/mouseslip/live/` | Feed, game state, Stockfish referee, watcher, HTTP server |
| `web/` | Static page served by the watcher |
| `scripts/` | Training script, watchdog, terminal demo |
| `results/` | Test results as JSON |
| `tests/` | 44 tests |

## Licence

MIT. The dataset keeps Lichess's CC0.
