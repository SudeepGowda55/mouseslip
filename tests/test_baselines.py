from dataclasses import replace
import pickle

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

from gardyloo.baselines import (
    FEATURE_SETS,
    BaselinePredictor,
    bootstrap_aucs,
    clock_dynamics,
    engine,
    features,
    matrix,
    move_times,
    priors,
    speed_bucket,
    top_precision,
)
from gardyloo.data import Window
from gardyloo.predict import BASELINE_MODEL, make_window


def test_move_times_without_increment():
    # white: 300 -> 290 -> 285, black: 300 -> 299 -> 279; white to move
    window = make_window([300, 300, 290, 299, 285, 279], "300+0", 1500, 1700)
    own, opponent = move_times(window)
    assert window.side_to_move == "white"
    assert own.tolist() == [10, 5]
    assert opponent.tolist() == [1, 20]


def test_move_times_add_the_increment_back():
    # 3 s increment: a clock that rises by 1 means the move took 2 s
    window = make_window([300, 300, 301, 298, 302], "300+3", 1500, 1700)
    own, opponent = move_times(window)
    assert window.side_to_move == "black"
    assert own.tolist() == [5]
    assert opponent.tolist() == [2, 2]
    assert clock_dynamics(window)["moves_left"] == pytest.approx(298 / (5 - 3))


def test_berserk_start_is_not_a_long_think():
    # white berserked a 300+3 game: starts on 150 and gets no increment
    window = make_window([150, 300, 148, 301, 147, 302], "300+3", 1500, 1700)
    own, opponent = move_times(window)
    assert own.tolist() == [2, 1]
    assert opponent.tolist() == [2, 2]
    dynamics = clock_dynamics(window)
    assert dynamics["longest"] == 2
    assert dynamics["left"] == 147
    assert dynamics["left_share"] == pytest.approx(147 / 420)
    assert dynamics["left_gap_share"] < 0
    # Lichess game HwSF8MXA at full precision: white berserked 180+2 and the first reading carries a lag credit
    own, opponent = move_times(make_window([90.03, 180.03, 88.51, 181.23, 87.55, 181.47], "180+2", 1635, 1817))
    assert own == pytest.approx([1.52, 0.96], abs=1e-4)  # float32 clocks
    assert opponent == pytest.approx([0.8, 1.76], abs=1e-4)


def test_priors_are_from_the_mover_side():
    window = make_window([60, 60, 59], "60+0", 1500, 1700)
    assert priors(window) == {"mover_elo": 1700, "elo_gap": 200, "log_base": np.log1p(60), "increment": 0, "ply": 3, "white": 0.0}


def test_feature_sets_do_not_leak_into_each_other():
    window = make_window([300, 300, 290, 299, 285, 279] * 2, "300+0", 1500, 1700)
    assert not {"mover_elo", "elo_gap", "ply"} & set(clock_dynamics(window))
    assert set(features(window, "clocks")) == set(priors(window)) | set(clock_dynamics(window))
    assert not set(engine(window)) & set(features(window, "clocks"))
    # what a live game lacks must not reach the sets a live game uses
    offline = replace(window, evaluation=window.evaluation + 5, label=1, prompt="p", rationale="r", game_id="g")
    for feature_set in ("priors", "clock-dynamics", "clocks"):
        assert features(offline, feature_set) == features(window, feature_set)
    assert features(offline, "clocks+engine") != features(window, "clocks+engine")
    assert matrix([window], "clocks+engine").shape == (1, len(features(window, "clocks+engine")))


def test_engine_features_take_the_mover_view():
    # 12 plies with black to move: black's own moves are the even positions, so position 10 is black's last move
    evaluation = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3], np.float32)
    clock = np.full(12, 100, np.float32)
    black = Window(clock, clock, evaluation, "300+0", 1500, 1500, "black", 13, "", "", "", 0)
    values = engine(black)
    assert values["eval"] == -3
    assert values["eval_swing"] == 3
    assert values["own_worst_drop"] > 0.3
    assert values["opponent_worst_drop"] == 0


def test_speed_bucket():
    names = [speed_bucket(make_window([1, 1], tc, 1500, 1500)) for tc in ("15+0", "120+1", "180+0", "300+3", "600+0", "900+15")]
    assert names == ["bullet", "bullet", "blitz", "blitz", "rapid", "classical"]


def test_metrics():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    assert top_precision(y, p, 0.1) == 1.0
    assert top_precision(y, p, 0.2) == 0.5
    # ties at the cut count at their shared rate, whatever order the data is in
    assert top_precision(y, np.zeros(10), 0.3) == top_precision(y[::-1], np.zeros(10), 0.3) == pytest.approx(0.2)
    assert top_precision(y, p, 0.5) == top_precision(y[::-1], p[::-1], 0.5) == pytest.approx(0.4)
    games = np.array(list("aabbccddee"))
    labels = np.array([1, 0] * 5)  # both classes in every game, so every resample has an AUC
    aucs = bootstrap_aucs(labels, p, games, resamples=20)
    assert np.array_equal(aucs, bootstrap_aucs(labels, p, games, resamples=20))  # same resamples for every model
    assert aucs.min() >= 0 and aucs.max() <= 1 and aucs.std() > 0


def test_baseline_predictor_on_a_live_window(tmp_path):
    rng = np.random.default_rng(0)
    windows = [make_window(sorted(rng.uniform(1, 300, 14), reverse=True), "300+0", 1500, 1600) for _ in range(40)]
    model = HistGradientBoostingClassifier(max_iter=3).fit(matrix(windows, "clocks"), np.arange(40) % 2)
    path = tmp_path / "model.pkl"
    path.write_bytes(pickle.dumps({"model": model, "threshold": 0.4}))
    predictor = BaselinePredictor(path)

    window = make_window([180, 180, 170, 178, 150, 177, 120, 176], "180+0", 1500, 1600, game_id="abc")
    assert 0 <= predictor.p_blunder(window) <= 1
    assert predictor.threshold == 0.4
    # white spent 10, 20 and 30 seconds: 20 s a move leaves 6 moves on a 2:00 clock
    assert predictor.explain(window) == (
        "White has 2:00 left against 2:56. Their last 3 moves took 20.0 seconds each, "
        "so the clock lasts about 6 more moves at this pace."
    )
    shortest = make_window([180, 180], "180+2", 1500, 1600)
    assert predictor.explain(shortest) == "White has 3:00 left against 3:00."
    assert 0 <= predictor.p_blunder(shortest) <= 1
    assert "about 1 more move at" in predictor.explain(make_window([15, 15, 12, 14, 9, 13, 6, 12, 3, 11], "15+0", 1500, 1600))
    assert "not a worry" in predictor.explain(make_window([180, 180, 181, 181, 182], "180+2", 1500, 1600))


@pytest.mark.skipif(not BASELINE_MODEL.exists(), reason="run `uv run python -m gardyloo.baselines` first")
def test_shipped_model_is_small_and_loads():
    assert BASELINE_MODEL.stat().st_size < 5_000_000
    predictor = BaselinePredictor(BASELINE_MODEL)
    assert set(FEATURE_SETS) == {"priors", "clock-dynamics", "clocks", "clocks+engine"}
    calm = make_window([600, 600] + [600 - k for k in range(1, 21)], "600+0", 2000, 2000)
    assert 0 <= predictor.p_blunder(calm) <= 1
