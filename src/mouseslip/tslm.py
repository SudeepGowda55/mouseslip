"""Fine-tune OpenTSLM-SP (frozen Gemma, trained time-series encoder and projector) to call chess blunders.

The model never sees the board. In mode "clocks+engine" it reads both clocks and the engine's evaluation
bar; in mode "clocks" (the ablation) only the clocks. It first writes what it sees in the series and
then answers, so the answer can rest on its own description.

    uv run python -m mouseslip.tslm train
    uv run python -m mouseslip.tslm eval
    uv run python -m mouseslip.tslm explain --n 5
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
import torch.nn.functional as F

from mouseslip.data import Window, load_windows

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS = ROOT / "checkpoints"
RESULTS = ROOT / "results"
LLM_ID = "google/gemma-3-270m"  # gated: needs a Hugging Face token with the Gemma licence accepted
MODES = ("clocks+engine", "clocks")
PLIES = 40
ANSWERS = (" no", " yes")  # indexed by label
CUE = " Answer:"  # ends the description; the answer is the next token
DESCRIPTION_TOKENS = 60  # more than the longest description needs
DESCRIPTION_WEIGHT = 3.0  # the description can be learned exactly, the answer only in probability
VALIDATION_WINDOWS = 1000


def game_seconds(time_control: str) -> float:
    """Estimate one player's clock time for a whole game: the base plus 40 increments."""
    base, increment = map(float, time_control.split("+"))
    return base + 40 * increment


def bucket(time_control: str) -> str:
    """Name the speed of a time control the way Lichess does."""
    seconds = game_seconds(time_control)
    return "bullet" if seconds < 180 else "blitz" if seconds < 480 else "rapid" if seconds < 1500 else "classical"


def _mover_view(window: Window) -> tuple[str, str, np.ndarray, np.ndarray, np.ndarray]:
    """Return the mover's and opponent's names and clocks, and the seconds each ply after the first took."""
    white = window.side_to_move == "white"
    mover, opponent = ("White", "Black") if white else ("Black", "White")
    own, other = (window.white_clock, window.black_clock) if white else (window.black_clock, window.white_clock)
    own, other = own[-PLIES:], other[-PLIES:]
    increment = float(window.time_control.split("+")[1])
    # One clock moves per ply, so the drop in their sum is the time that ply took.
    # shortcut: a berserked player gets no increment, so their moves read one increment too long;
    # fix when the data says who berserked.
    spent = np.clip(increment - np.diff(own + other), 0, None)
    return mover, opponent, own, other, spent


def _left_pad(values: np.ndarray, fill: float | None = None) -> np.ndarray:
    # Right-aligned on the decision point, so the last patch always holds the latest moves.
    fill = values[0] if fill is None else fill
    return np.concatenate([np.full(PLIES - len(values), fill), values]).astype(np.float32)


def _chances(window: Window) -> np.ndarray:
    """Return the mover's winning chances after each ply of the window, on Lichess's -1 to +1 scale."""
    pawns = window.evaluation[-PLIES:] * (1 if window.side_to_move == "white" else -1)
    return 2 / (1 + np.exp(-0.368208 * pawns)) - 1


def sample(window: Window, mode: str = MODES[0]) -> dict:
    """Build the OpenTSLM prompt for a window: text plus series, with nothing from the label or the future.

    The text is terse because every prompt token is paid for in each step through the frozen LLM.
    The series are scaled to order one because the encoder adds unit-variance position embeddings to them.
    A clock is given as a share of the expected game time, not of the base: a berserked player starts
    at half the base and a "0+1" game has no base at all. Each series is introduced with its latest value
    in words, the way OpenTSLM's own prompts state a series' mean and spread.
    """
    mover, opponent, own, other, spent = _mover_view(window)
    seconds = game_seconds(window.time_control)
    series = {
        f"{mover} clock, {own[-1] / seconds:.0%} of the game time left:": _left_pad(own / seconds),
        f"{opponent} clock, {other[-1] / seconds:.0%} left:": _left_pad(other / seconds),
        f"{mover} log seconds:": _left_pad(np.log10(1 + own)),
        "Time per ply:": _left_pad(np.clip(spent / (seconds / 40), 0, 10), 0),  # in average-move units
    }
    if mode == "clocks+engine":
        chances = _chances(window)
        series[f"Winning chances for {mover}, now {chances[-1]:+.2f}:"] = _left_pad(chances)
    shown = "Chess clocks and evaluation bar" if mode == "clocks+engine" else "Chess clocks"
    return {
        "pre_prompt": (
            f"{shown}, board hidden. Time control {window.time_control}. "
            f"White {window.white_elo}, Black {window.black_elo}. {mover} to move at ply {window.ply}."
        ),
        "time_series_text": list(series),
        "time_series": list(series.values()),
        "post_prompt": f"Will {mover} blunder within three moves? Describe the series, then answer.",
    }


def reason(window: Window, mode: str = "clocks") -> str:
    """Describe the mover's situation in plain words, from the model's own inputs only, never the outcome."""
    mover, opponent, own, other, spent = _mover_view(window)
    seconds = game_seconds(window.time_control)
    share, lead = own[-1] / seconds, (own[-1] - other[-1]) / seconds
    pace = spent[-2::-2][:3] / (seconds / 40)  # the mover's last three moves, latest first
    left = "is almost out of time" if share < 0.1 else "is low on time" if share < 0.3 else "has enough time"
    gap = "is behind" if lead < -0.1 else "is ahead" if lead > 0.1 else "is level"
    if pace.size and pace[0] > 2:
        tempo = "after a long think"
    elif pace.size and pace.mean() < 0.34:
        tempo = "moving fast"
    else:
        tempo = "at a steady pace"
    text = f"{mover} {left}, {gap} on the clock, {tempo}."
    if mode == "clocks+engine":
        chances = _chances(window)
        now, swing = chances[-1], float(np.ptp(chances[-10:]))
        standing = (
            "is winning" if now > 0.6 else "is better" if now > 0.2 else "is losing" if now < -0.6
            else "is worse" if now < -0.2 else "is about level"
        )  # fmt: skip
        mood = "swinging wildly" if swing > 0.6 else "shifting" if swing > 0.25 else "steady"
        # Each ply changed the mover's chances; the last ply was the opponent's, the one before it the mover's.
        change = np.diff(chances)
        own_blunders, their_blunders = int((-change[-2::-2] >= 0.3).sum()), int((change[-1::-2] >= 0.3).sum())
        text += (
            f" Board: {mover} {standing}, evaluation {mood}."
            f" Blunders: {mover} {own_blunders}, {opponent} {their_blunders}."
        )
    return text


def target(window: Window, mode: str = "clocks") -> str:
    """Return the text the model learns to write: what the series show, then the answer."""
    return f" {reason(window, mode)}{CUE}{ANSWERS[window.label]}"


def subset(windows: list[Window], limit: int | None, seed: int = 0) -> list[Window]:
    """Return a fixed random subset: neighbouring windows come from the same game, so a head slice is no sample."""
    if limit is None or limit >= len(windows):
        return windows
    return [windows[i] for i in np.random.default_rng(seed).choice(len(windows), limit, replace=False)]


def batch_indices(n: int, batch_size: int, step: int, seed: int) -> np.ndarray:
    """Return the windows of a training step; a pure function of the step, so a resumed run keeps the order."""
    epoch, index = divmod(step, n // batch_size)
    return np.random.default_rng(seed + epoch).permutation(n)[index * batch_size : (index + 1) * batch_size]


def best_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """Return the score cut-off with the highest accuracy; 2.0 means never calling a blunder is best."""
    candidates = np.append(np.unique(scores), 2.0)
    return float(max(candidates, key=lambda cut: np.mean((scores >= cut) == labels)))


def metrics(windows: list[Window], scores: np.ndarray, threshold: float) -> dict:
    """Score predictions against the labels, overall and per speed bucket."""
    labels = np.array([w.label for w in windows])
    speeds = np.array([bucket(w.time_control) for w in windows])

    def ranking(keep: np.ndarray) -> dict:
        y, s = labels[keep], scores[keep]
        both = 0 < y.sum() < len(y)
        return {
            "n": int(len(y)),
            "base_rate": float(y.mean()),
            "auc": float(roc_auc_score(y, s)) if both else None,
            "average_precision": float(average_precision_score(y, s)) if both else None,
        }

    top = np.argsort(-scores)[: max(1, len(scores) // 10)]
    base_rate = float(labels.mean())
    rng = np.random.default_rng(0)
    resamples = (rng.integers(0, len(labels), len(labels)) for _ in range(1000))
    aucs = [roc_auc_score(labels[i], scores[i]) for i in resamples if 0 < labels[i].sum() < len(i)]
    return {
        **ranking(np.ones(len(labels), bool)),
        "auc_ci95": [float(x) for x in np.percentile(aucs, [2.5, 97.5])],  # bootstrap over the scored windows
        "top_decile_precision": float(labels[top].mean()),
        "top_decile_lift": float(labels[top].mean() / base_rate) if base_rate else None,
        "threshold": threshold,
        "accuracy": float(np.mean((scores >= threshold) == labels)),
        "majority_class_accuracy": max(base_rate, 1 - base_rate),
        "by_speed": {name: ranking(speeds == name) for name in ("bullet", "blitz", "rapid", "classical") if (speeds == name).any()},
    }


def default_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_model(device: str, checkpoint: dict | None = None, *, lora: bool = False, llm_id: str = LLM_ID):
    """Build OpenTSLM-SP on the frozen LLM and load trained encoder, projector and LoRA weights if given."""
    from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP  # noqa: PLC0415 (slow import, kept out of the tests)

    model = OpenTSLMSP(llm_id=checkpoint["llm_id"] if checkpoint else llm_id, device=device)
    model.llm.float()  # bfloat16 is slow or broken on MPS
    if lora or (checkpoint and checkpoint.get("lora_enabled")):
        model.enable_lora()
    if checkpoint:
        model.encoder.load_state_dict(checkpoint["encoder_state"])
        model.projector.load_state_dict(checkpoint["projector_state"])
        model.load_lora_state_from_checkpoint(checkpoint, allow_missing=True)
    # No special tokens: Gemma's tokenizer would put <bos> first, and the first token must be the answer.
    model.answer_ids = [model.tokenizer(answer, add_special_tokens=False).input_ids[0] for answer in ANSWERS]
    assert model.answer_ids[0] != model.answer_ids[1], "yes and no must differ at the first answer token"
    in_context = model.tokenizer(f"x{CUE}{ANSWERS[1]}", add_special_tokens=False).input_ids
    assert in_context[-1] == model.answer_ids[1], "after the cue the answer must still be one token"
    return model


def _rows(model, windows: list[Window], mode: str, texts: list[str] | None = None):
    """Embed each window's prompt, followed by its text if given, padded on the left.

    Every row then ends at the last position, in training exactly as when the model writes on its own.

    Returns:
        The embeddings, the attention mask, and the token ids of the appended texts aligned to the right
        (-100 where a row has none).
    """
    from opentslm.time_series_datasets.util import (  # noqa: PLC0415
        extend_time_series_to_match_patch_size_and_aggregate as collate,
    )

    embeds, mask = model.pad_and_apply_batch(collate([sample(w, mode) for w in windows]))
    rows = [embeds[i, :n] for i, n in enumerate(mask.sum(1).tolist())]  # OpenTSLM pads on the right
    ids = []
    if texts is not None:
        ids = [torch.tensor(model.tokenizer(t, add_special_tokens=False).input_ids, device=embeds.device) for t in texts]
        rows = [torch.cat([row, model.llm.get_input_embeddings()(i).to(row.dtype)]) for row, i in zip(rows, ids, strict=True)]
    width = max(len(row) for row in rows)
    padded = embeds.new_zeros(len(rows), width, embeds.size(-1))
    kept = mask.new_zeros(len(rows), width)
    labels = torch.full((len(rows), width), -100, dtype=torch.long, device=embeds.device)
    for k, row in enumerate(rows):
        padded[k, width - len(row) :], kept[k, width - len(row) :] = row, 1
        if ids:
            labels[k, width - len(ids[k]) :] = ids[k]
    return padded, kept, labels


def _hidden(model, embeds: torch.Tensor, mask: torch.Tensor):
    """Run the LLM without its output layer and return it with the hidden states.

    Gemma's vocabulary has 262,144 entries, so logits at every position cost four times the rest of a step;
    the callers apply the output layer only where a token is read.
    """
    llm = model.llm.get_base_model() if model.lora_enabled else model.llm
    return llm, llm.model(inputs_embeds=embeds, attention_mask=mask).last_hidden_state


def losses(model, windows: list[Window], mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the cross-entropy of the yes/no token and the mean cross-entropy of the description tokens.

    They are kept apart because in a plain mean over the target the yes/no token, the only one that
    carries the prediction, would be one token in about fifty.
    """
    embeds, mask, labels = _rows(model, windows, mode, [target(w, mode) + model.tokenizer.eos_token for w in windows])
    llm, hidden = _hidden(model, embeds, mask)
    n = int((labels != -100).sum(1).max())
    labels = labels[:, -n:]
    logits = llm.lm_head(hidden[:, -n - 1 : -1]).float()  # the predictions for the last n positions
    nll = F.cross_entropy(logits.transpose(1, 2), labels, reduction="none", ignore_index=-100)
    described = labels != -100
    described[:, -2:] = False  # every row ends with the answer and the end token
    return nll[:, -2].mean(), (nll * described).sum() / described.sum()


@torch.no_grad()
def describe(model, windows: list[Window], mode: str) -> list[str]:
    """Let the model write what it sees in each window; the text stops before the answer cue."""
    embeds, mask, _ = _rows(model, windows, mode)
    ids = model.llm.generate(
        inputs_embeds=embeds,
        attention_mask=mask,
        max_new_tokens=DESCRIPTION_TOKENS,
        do_sample=False,
        pad_token_id=model.tokenizer.pad_token_id,
    )
    return [text.split(CUE.strip())[0].strip() for text in model.tokenizer.batch_decode(ids, skip_special_tokens=True)]


@torch.no_grad()
def score(model, windows: list[Window], mode: str, batch_size: int = 16, *, own_words: bool = True):
    """Return P(blunder) and the description for each window.

    P is "yes" against "no" right after the description. With `own_words` the model writes the description
    itself, as it must on a live game. Without, the answer is read after the reference description: one
    forward pass, used during training to pick the checkpoint, and reported apart from the real score.
    """
    model.eval()
    scores, descriptions = [], []
    for start in range(0, len(windows), batch_size):
        chunk = windows[start : start + batch_size]
        texts = describe(model, chunk, mode) if own_words else [reason(w, mode) for w in chunk]
        embeds, mask, _ = _rows(model, chunk, mode, [f" {text}{CUE}" for text in texts])
        llm, hidden = _hidden(model, embeds, mask)
        logits = llm.lm_head(hidden[:, -1])[:, model.answer_ids]
        scores.append(logits.float().softmax(-1)[:, 1].cpu().numpy())
        descriptions += texts
    return np.concatenate(scores), descriptions


class TslmPredictor:
    """The fine-tuned model behind the `Predictor` interface."""

    def __init__(self, checkpoint_path: str | Path, device: str | None = None) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.mode = checkpoint["mode"]
        self.name = f"OpenTSLM-SP on {checkpoint['llm_id'].split('/')[1]} ({self.mode}, {checkpoint['step']} steps)"
        self.model = load_model(device or default_device(), checkpoint)

        self._last: tuple[Window, float, str] | None = None

    def _read(self, window: Window) -> tuple[Window, float, str]:
        # The description and the probability come from one pass, and the watcher asks for them one after the other.
        if self._last is None or self._last[0] is not window:
            scores, descriptions = score(self.model, [window], self.mode)
            self._last = (window, float(scores[0]), descriptions[0])
        return self._last

    def p_blunder(self, window: Window) -> float:
        return self._read(window)[1]

    def explain(self, window: Window) -> str:
        return self._read(window)[2]


def train(args: argparse.Namespace) -> None:
    from opentslm.model_config import GRAD_CLIP_NORM, LR_ENCODER, LR_PROJECTOR, WARMUP_FRAC, WEIGHT_DECAY  # noqa: PLC0415
    from transformers import get_linear_schedule_with_warmup  # noqa: PLC0415

    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    last, best_path = args.out / f"last-{args.mode}.pt", args.out / f"tslm-{args.mode}.pt"
    state = torch.load(last, map_location="cpu", weights_only=False) if args.resume else None
    model = load_model(args.device, state, lora=args.lora, llm_id=args.llm)

    groups = [
        {"params": list(model.encoder.parameters()), "lr": LR_ENCODER},
        {"params": list(model.projector.parameters()), "lr": LR_PROJECTOR},
    ]
    if model.lora_enabled:
        groups.append({"params": model.get_lora_parameters(), "lr": LR_PROJECTOR})
    trainable = [p for group in groups for p in group["params"]]
    optimizer = torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)

    windows = load_windows("train")
    validation = subset(load_windows("validation"), VALIDATION_WINDOWS)
    validation_labels = np.array([w.label for w in validation])
    total = args.epochs * (len(windows) // args.batch_size)
    total = min(total, args.max_steps) if args.max_steps else total
    scheduler = get_linear_schedule_with_warmup(optimizer, int(WARMUP_FRAC * total), total)
    step, best = 0, -1.0
    if state:
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step, best = state["step"], state["best_val_auc"]

    with (args.out / f"train-{args.mode}.log").open("a" if args.resume else "w") as log:

        def write(**record) -> None:
            log.write(json.dumps({"step": step, **record}) + "\n")
            log.flush()
            print(json.dumps({"step": step, **record}), flush=True)

        recent, started = [], time.time()
        while step < total:
            model.train()
            batch = [windows[i] for i in batch_indices(len(windows), args.batch_size, step, args.seed)]
            answer_loss, reason_loss = losses(model, batch, args.mode)
            (answer_loss + DESCRIPTION_WEIGHT * reason_loss).backward()
            torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            recent.append((answer_loss.item(), reason_loss.item()))

            if step % args.log_every == 0 or step == total:
                answer_mean, reason_mean = np.mean(recent, axis=0)
                write(
                    loss=round(answer_mean + reason_mean, 4),
                    loss_answer=round(answer_mean, 4),
                    loss_reason=round(reason_mean, 4),
                    lr=scheduler.get_last_lr()[0],
                    seconds_per_step=round((time.time() - started) / len(recent), 3),
                )
                recent, started = [], time.time()

            if step % args.every == 0 or step == total:
                scores, _ = score(model, validation, args.mode, own_words=False)
                auc = float(roc_auc_score(validation_labels, scores))
                weights = {
                    "encoder_state": model.encoder.state_dict(),
                    "projector_state": model.projector.state_dict(),
                    "llm_id": args.llm,
                    "mode": args.mode,
                    "step": step,
                    "val_auc": auc,
                    "threshold": best_threshold(validation_labels, scores),
                }
                model.save_lora_state_to_checkpoint(weights)  # sets lora_enabled, adds the adapters if any
                if auc > best:
                    best = auc
                    torch.save(weights, best_path)
                torch.save(
                    {**weights, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "best_val_auc": best},
                    last,
                )
                # How well it reads the series on its own: slow, so on a few windows only.
                sample_windows = validation[:64]
                _, written = score(model, sample_windows, args.mode)
                match = float(np.mean([text == reason(w, args.mode) for text, w in zip(written, sample_windows, strict=True)]))
                write(val_auc=round(auc, 4), best_val_auc=round(best, 4), val_description_match=round(match, 3))
                started = time.time()  # keep validation time out of seconds_per_step


def evaluate(args: argparse.Namespace) -> None:
    checkpoint = torch.load(args.checkpoint or CHECKPOINTS / f"tslm-{args.mode}.pt", map_location="cpu", weights_only=False)
    mode = checkpoint["mode"]
    model = load_model(args.device, checkpoint)
    windows = subset(load_windows("test"), args.limit)
    scores, descriptions = score(model, windows, mode)
    reference_scores, _ = score(model, windows, mode, own_words=False)
    labels = np.array([w.label for w in windows])
    result = {
        "model": f"OpenTSLM-SP on {checkpoint['llm_id']}" + (" + LoRA" if checkpoint.get("lora_enabled") else ""),
        "mode": mode,
        "split": "test",
        "steps_trained": checkpoint["step"],
        "validation_auc": checkpoint["val_auc"],
        "device": args.device,
        "answers_after": "the model's own description of the series",
        **metrics(windows, scores, checkpoint["threshold"]),
        # How well it reads the series: its description against the one computed from the same numbers.
        "description_exact_match": float(np.mean([d == reason(w, mode) for d, w in zip(descriptions, windows, strict=True)])),
        "auc_after_reference_description": float(roc_auc_score(labels, reference_scores)),
        "call_threshold_top_decile": float(np.quantile(scores, 0.9)),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"tslm-{mode}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def show(args: argparse.Namespace) -> None:
    checkpoint_path = args.checkpoint or CHECKPOINTS / f"tslm-{args.mode}.pt"
    predictor = TslmPredictor(checkpoint_path, args.device)
    print(predictor.name)
    for window in subset(load_windows("test"), args.n, args.seed):
        print(
            f"\n{window.game_id} {window.time_control} ply {window.ply}, {window.side_to_move} to move\n"
            f"  P(blunder) {predictor.p_blunder(window):.3f}   truth: {ANSWERS[window.label].strip()}\n"
            f"  generated: {predictor.explain(window)}\n"
            f"  reference: {reason(window, predictor.mode)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    def command(name: str, run) -> argparse.ArgumentParser:
        sub = commands.add_parser(name)
        sub.set_defaults(run=run)
        sub.add_argument("--mode", choices=MODES, default=MODES[0])
        sub.add_argument("--device", default=default_device())
        return sub

    sub = command("train", train)
    sub.add_argument("--epochs", type=int, default=1)
    sub.add_argument("--max-steps", type=int, help="stop early; the learning-rate schedule spans this many steps")
    sub.add_argument("--batch-size", type=int, default=4)
    sub.add_argument("--every", type=int, default=500, help="steps between validations and checkpoints")
    sub.add_argument("--log-every", type=int, default=20)
    sub.add_argument("--seed", type=int, default=0)
    sub.add_argument("--resume", action="store_true", help="continue from last-<mode>.pt; pass the same other flags")
    sub.add_argument("--lora", action="store_true", help="also train LoRA adapters in the LLM")
    sub.add_argument("--llm", default=LLM_ID, help="the frozen base model, a Hugging Face id")
    sub.add_argument("--out", type=Path, default=CHECKPOINTS, help="directory for checkpoints and the log")

    sub = command("eval", evaluate)
    sub.add_argument("--checkpoint", type=Path)
    sub.add_argument("--limit", type=int, help="score a fixed random subset of the test split")

    sub = command("explain", show)
    sub.add_argument("--checkpoint", type=Path)
    sub.add_argument("--n", type=int, default=5)
    sub.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
