"""Sample building, target format and metrics; nothing here loads a model."""

from dataclasses import replace

import numpy as np
import pytest

from mouseslip import tslm
from mouseslip.predict import make_window


def window(clocks, time_control="300+0", label=0):
    return replace(make_window(clocks, time_control, 1800, 1700), label=label)


def low_on_time(label=0):
    # 30 plies of a 300+0 game: White burns 20 s a move, Black 2 s, so White is to move with 0 s left.
    clocks = [300 - (20 if ply % 2 == 0 else 2) * (ply // 2 + 1) for ply in range(30)]
    return window(clocks, label=label)


@pytest.mark.parametrize("mode", tslm.MODES)
def test_sample_shape_and_scale(mode):
    sample = tslm.sample(low_on_time(), mode)
    assert len(sample["time_series"]) == len(sample["time_series_text"]) == (4 if mode == "clocks" else 5)
    for series in sample["time_series"]:
        assert series.shape == (tslm.PLIES,) and series.dtype == np.float32
        assert np.isfinite(series).all() and np.abs(series).max() <= 10
    own, other, _, spent = sample["time_series"][:4]
    assert own[-1] == 0 and other[-1] == pytest.approx(270 / 300)
    assert own[:10] == pytest.approx(own[10])  # short games are padded on the left with the first reading
    assert spent[-2:] == pytest.approx([20 / 7.5, 2 / 7.5]) and (spent[:11] == 0).all()


def test_mover_is_black_on_odd_ply():
    sample = tslm.sample(window([300, 300, 290, 299, 280]))
    assert sample["time_series_text"][0].startswith("Black clock")
    assert sample["time_series"][0][-1] == pytest.approx(299 / 300)
    assert "Black to move at ply 5" in sample["pre_prompt"]


def test_increment_and_zero_base_time_controls():
    spent = tslm.sample(window([60, 60, 58, 61], "60+2"))["time_series"][3]
    assert spent[-2:] == pytest.approx([4 / 3.5, 1 / 3.5])  # 2 s increment back: 4 s and 1 s thinks
    assert np.isfinite(tslm.sample(window([3, 3, 2, 4], "0+1"))["time_series"][0]).all()


def test_prompt_and_reason_do_not_depend_on_the_label():
    no = replace(low_on_time(0), rationale="None of the next three moves is a blunder. Answer: no")
    yes = replace(low_on_time(1), rationale="White blunders on the next move. Answer: yes")
    for mode in tslm.MODES:
        a, b = tslm.sample(no, mode), tslm.sample(yes, mode)
        assert all(np.array_equal(x, y) for x, y in zip(a.pop("time_series"), b.pop("time_series"), strict=True))
        assert a == b
        text = " ".join([a["pre_prompt"], *a["time_series_text"], a["post_prompt"]]).lower()
        assert "answer: " not in text and "yes" not in text and "rationale" not in text
    assert tslm.reason(no) == tslm.reason(yes)


def test_clocks_mode_ignores_the_evaluation():
    quiet = low_on_time()
    lost = replace(quiet, evaluation=np.full_like(quiet.evaluation, -9.0))
    a, b = tslm.sample(quiet, "clocks"), tslm.sample(lost, "clocks")
    assert all(np.array_equal(x, y) for x, y in zip(a["time_series"], b["time_series"], strict=True))
    engine = tslm.sample(lost, "clocks+engine")["time_series"][4]
    assert engine[-1] < -0.9  # White is to move and lost


def test_target_is_description_then_answer():
    assert tslm.target(low_on_time(1)) == (
        " White is almost out of time, is behind on the clock, after a long think. Answer: yes"
    )
    assert tslm.target(low_on_time(0)).endswith("after a long think. Answer: no")
    fast = window([300 - ply // 2 for ply in range(24)], label=0)
    assert tslm.target(fast) == " White has enough time, is level on the clock, moving fast. Answer: no"


def test_engine_description_counts_the_blunders_already_in_view():
    game = low_on_time()  # 30 plies, White to move
    game.evaluation[:] = 0.0
    game.evaluation[-2:] = -3.0  # White's last move lost the game; Black's reply kept it
    assert tslm.reason(game, "clocks+engine").endswith("Blunders: White 1, Black 0.")
    game.evaluation[-1] = 0.0  # ...and Black's reply gave it straight back
    assert tslm.reason(game, "clocks+engine").endswith("Blunders: White 1, Black 1.")


def test_works_on_the_shortest_live_window():
    shortest = make_window([300, 300], "300+0", 1500, 1500)
    assert tslm.sample(shortest)["time_series"][0].shape == (tslm.PLIES,)
    assert tslm.reason(shortest).endswith("at a steady pace.")


@pytest.mark.parametrize(
    ("time_control", "name"),
    [("15+0", "bullet"), ("60+2", "bullet"), ("120+1", "bullet"), ("180+0", "blitz"), ("300+3", "blitz"),
     ("300+5", "rapid"), ("600+0", "rapid"), ("900+15", "classical"), ("1800+0", "classical"), ("0+1", "bullet")],
)  # fmt: skip
def test_bucket(time_control, name):
    assert tslm.bucket(time_control) == name


def test_batch_indices_cover_an_epoch_and_reshuffle():
    first = np.concatenate([tslm.batch_indices(10, 4, step, seed=0) for step in range(2)])
    assert len(set(first)) == 8 and (first == np.concatenate([tslm.batch_indices(10, 4, s, 0) for s in range(2)])).all()
    assert not np.array_equal(first[:4], tslm.batch_indices(10, 4, 2, seed=0))  # step 2 opens the second epoch


def test_subset_is_fixed_and_bounded():
    items = list(range(50))
    assert tslm.subset(items, 10) == tslm.subset(items, 10) and len(set(tslm.subset(items, 10))) == 10
    assert tslm.subset(items, None) is items and tslm.subset(items, 99) is items


def test_metrics_and_threshold():
    windows = [window([300, 300, 290, 290], tc, label) for tc, label in [("60+0", 1), ("60+0", 0), ("600+0", 1), ("600+0", 0)] * 5]
    perfect = np.array([0.9, 0.1, 0.8, 0.2] * 5)
    labels = np.array([w.label for w in windows])
    cut = tslm.best_threshold(labels, perfect)
    result = tslm.metrics(windows, perfect, cut)
    assert result["auc_ci95"] == [1.0, 1.0]  # a perfect ranking stays perfect under resampling
    assert cut == pytest.approx(0.8) and result["accuracy"] == 1 and result["auc"] == 1 and result["n"] == 20
    assert result["base_rate"] == 0.5 and result["top_decile_precision"] == 1 and result["top_decile_lift"] == 2
    assert set(result["by_speed"]) == {"bullet", "rapid"} and result["by_speed"]["rapid"]["n"] == 10
    assert tslm.best_threshold(np.array([0, 0, 0, 1]), np.array([0.9, 0.2, 0.3, 0.1])) == 2.0  # never calling wins


def test_engine_reason_reads_the_evaluation_and_not_the_label():
    winning, losing = low_on_time(0), low_on_time(1)
    losing.evaluation[:] = -6.0  # White to move and far behind
    assert tslm.reason(winning, "clocks+engine") != tslm.reason(winning, "clocks")
    assert "is losing" in tslm.reason(losing, "clocks+engine")
    same = low_on_time(1)
    assert tslm.reason(same, "clocks+engine") == tslm.reason(low_on_time(0), "clocks+engine")
