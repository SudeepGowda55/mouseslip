"""Offline tests for the live watcher's pure logic: no network, no engine process."""

import json

import chess
import chess.engine
import pytest

from gongoozler.live.game import Decision, Game, Scoreboard, chances_lost, winning_chances
from gongoozler.live.heuristic import DevHeuristicPredictor
from gongoozler.live.verifier import score_chances, score_pawns, terminal_chances
from gongoozler.live.watcher import Watcher

# One real rapid game as /api/stream/game/<id> sent it (White resigned after 32 plies); id and names replaced.
RECORDED = """\
{"id":"sample01","variant":{"key":"standard","name":"Standard","short":"Std"},"speed":"rapid","perf":"rapid","rated":true,"players":{"white":{"user":{"name":"Manvith","title":"FM","id":"manvith"},"rating":2376},"black":{"user":{"name":"Twisha","id":"twisha"},"rating":2530}}}
{"fen":"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1","wc":600,"bc":600}
{"fen":"rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1","lm":"d2d4","wc":600,"bc":600}
{"fen":"rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2","lm":"d7d5","wc":600,"bc":600}
{"fen":"rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2","lm":"c2c4","wc":599,"bc":600}
{"fen":"rnbqkbnr/pp2pppp/2p5/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3","lm":"c7c6","wc":599,"bc":598}
{"fen":"rnbqkbnr/pp2pppp/2p5/3p4/2PP4/5N2/PP2PPPP/RNBQKB1R b KQkq - 1 3","lm":"g1f3","wc":598,"bc":598}
{"fen":"rnbqkb1r/pp2pppp/2p2n2/3p4/2PP4/5N2/PP2PPPP/RNBQKB1R w KQkq - 2 4","lm":"g8f6","wc":598,"bc":596}
{"fen":"rnbqkb1r/pp2pppp/2p2n2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R b KQkq - 3 4","lm":"b1c3","wc":598,"bc":596}
{"fen":"rnbqkb1r/pp2pppp/2p2n2/8/2pP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 0 5","lm":"d5c4","wc":598,"bc":596}
{"fen":"rnbqkb1r/pp2pppp/2p2n2/8/P1pP4/2N2N2/1P2PPPP/R1BQKB1R b KQkq - 0 5","lm":"a2a4","wc":596,"bc":596}
{"fen":"rn1qkb1r/pp2pppp/2p2n2/5b2/P1pP4/2N2N2/1P2PPPP/R1BQKB1R w KQkq - 1 6","lm":"c8f5","wc":596,"bc":595}
{"fen":"rn1qkb1r/pp2pppp/2p2n2/5b2/P1pP4/2N1PN2/1P3PPP/R1BQKB1R b KQkq - 0 6","lm":"e2e3","wc":595,"bc":595}
{"fen":"rn1qkb1r/pp3ppp/2p1pn2/5b2/P1pP4/2N1PN2/1P3PPP/R1BQKB1R w KQkq - 0 7","lm":"e7e6","wc":595,"bc":593}
{"fen":"rn1qkb1r/pp3ppp/2p1pn2/5b2/P1BP4/2N1PN2/1P3PPP/R1BQK2R b KQkq - 0 7","lm":"f1c4","wc":594,"bc":593}
{"fen":"rn1qk2r/pp3ppp/2p1pn2/5b2/PbBP4/2N1PN2/1P3PPP/R1BQK2R w KQkq - 1 8","lm":"f8b4","wc":594,"bc":593}
{"fen":"rn1qk2r/pp3ppp/2p1pn2/5b2/PbBP4/2N1PN2/1P3PPP/R1BQ1RK1 b kq - 2 8","lm":"e1g1","wc":593,"bc":593}
{"fen":"r2qk2r/pp1n1ppp/2p1pn2/5b2/PbBP4/2N1PN2/1P3PPP/R1BQ1RK1 w kq - 3 9","lm":"b8d7","wc":593,"bc":593}
{"fen":"r2qk2r/pp1n1ppp/2p1pn2/5b2/PbBP4/2N1PN2/1P2QPPP/R1B2RK1 b kq - 4 9","lm":"d1e2","wc":592,"bc":593}
{"fen":"r2q1rk1/pp1n1ppp/2p1pn2/5b2/PbBP4/2N1PN2/1P2QPPP/R1B2RK1 w - - 5 10","lm":"e8g8","wc":592,"bc":590}
{"fen":"r2q1rk1/pp1n1ppp/2p1pn2/5b2/PbBPP3/2N2N2/1P2QPPP/R1B2RK1 b - - 0 10","lm":"e3e4","wc":591,"bc":590}
{"fen":"r2q1rk1/pp1n1ppp/2p1pnb1/8/PbBPP3/2N2N2/1P2QPPP/R1B2RK1 w - - 1 11","lm":"f5g6","wc":591,"bc":590}
{"fen":"r2q1rk1/pp1n1ppp/2p1pnb1/4P3/PbBP4/2N2N2/1P2QPPP/R1B2RK1 b - - 0 11","lm":"e4e5","wc":587,"bc":590}
{"fen":"r2q1rk1/pp1n1ppp/2p1p1b1/3nP3/PbBP4/2N2N2/1P2QPPP/R1B2RK1 w - - 1 12","lm":"f6d5","wc":587,"bc":589}
{"fen":"r2q1rk1/pp1n1ppp/2p1p1b1/3nP3/PbBPN3/5N2/1P2QPPP/R1B2RK1 b - - 2 12","lm":"c3e4","wc":581,"bc":589}
{"fen":"r2q1rk1/pp1nbppp/2p1p1b1/3nP3/P1BPN3/5N2/1P2QPPP/R1B2RK1 w - - 3 13","lm":"b4e7","wc":581,"bc":546}
{"fen":"r2q1rk1/pp1nbppp/2p1p1b1/3nP1B1/P1BPN3/5N2/1P2QPPP/R4RK1 b - - 4 13","lm":"c1g5","wc":575,"bc":546}
{"fen":"r2q1rk1/pp1nbppp/2p1p3/3nP1B1/P1BPb3/5N2/1P2QPPP/R4RK1 w - - 0 14","lm":"g6e4","wc":575,"bc":537}
{"fen":"r2q1rk1/pp1nBppp/2p1p3/3nP3/P1BPb3/5N2/1P2QPPP/R4RK1 b - - 0 14","lm":"g5e7","wc":566,"bc":537}
{"fen":"r2q1rk1/pp1nBppp/2p1p3/3nP3/P1BP4/5b2/1P2QPPP/R4RK1 w - - 0 15","lm":"e4f3","wc":566,"bc":536}
{"fen":"r2B1rk1/pp1n1ppp/2p1p3/3nP3/P1BP4/5b2/1P2QPPP/R4RK1 b - - 0 15","lm":"e7d8","wc":563,"bc":536}
{"fen":"r2B1rk1/pp1n1ppp/2p1p3/3nP3/P1BP4/8/1P2bPPP/R4RK1 w - - 0 16","lm":"f3e2","wc":563,"bc":534}
{"fen":"r2B1rk1/pp1n1ppp/2p1p3/3nP3/P2P4/8/1P2BPPP/R4RK1 b - - 0 16","lm":"c4e2","wc":561,"bc":534}
{"fen":"r2r2k1/pp1n1ppp/2p1p3/3nP3/P2P4/8/1P2BPPP/R4RK1 w - - 0 17","lm":"f8d8","wc":561,"bc":532}
{"id":"sample01","variant":{"key":"standard","name":"Standard","short":"Std"},"speed":"rapid","perf":"rapid","rated":true,"fen":"r2r2k1/pp1n1ppp/2p1p3/3nP3/P2P4/8/1P2BPPP/R4RK1 w - - 0 17","turns":32,"status":{"id":31,"name":"resign"},"winner":"black","players":{"white":{"user":{"name":"Manvith","title":"FM","id":"manvith"},"rating":2376,"ratingDiff":-4},"black":{"user":{"name":"Twisha","id":"twisha"},"rating":2530,"ratingDiff":3}}}
"""
def replay(lines: int | None = None) -> Game:
    game = Game("sample01", "rapid", "600+0")
    for line in RECORDED.splitlines()[:lines]:
        game.feed(json.loads(line))
    return game


def decide(game: Game, ply: int, *, called: bool = True) -> Decision:
    game.open[ply] = Decision(ply, "white" if ply % 2 == 0 else "black", 0.5, called)
    return game.open[ply]


def test_winning_chances_follow_the_lichess_formula():
    assert winning_chances(0) == 0
    assert winning_chances(100) == pytest.approx(0.1820, abs=1e-4)
    assert winning_chances(-100) == pytest.approx(-winning_chances(100))
    assert winning_chances(1e9) == pytest.approx(1)  # no overflow on absurd scores
    assert chances_lost(0.5, 0.1, white_moved=True) == pytest.approx(0.4)
    assert chances_lost(0.5, 0.1, white_moved=False) == pytest.approx(-0.4)


def test_engine_scores_become_winning_chances():
    assert score_chances(chess.engine.Mate(3)) == 1.0
    assert score_chances(chess.engine.Mate(-1)) == -1.0
    assert score_chances(chess.engine.Cp(-100)) == pytest.approx(-0.1820, abs=1e-4)
    mated = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert terminal_chances(mated) == -1.0  # White is checkmated
    assert terminal_chances(chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")) == 0.0  # stalemate
    assert terminal_chances(chess.Board()) is None


def test_engine_scores_become_the_pawns_the_model_reads():
    assert score_pawns(chess.engine.Mate(2)) == 10.0 and score_pawns(chess.engine.Mate(-4)) == -10.0
    assert score_pawns(chess.engine.Cp(-135)) == -1.35 and score_pawns(chess.engine.Cp(2500)) == 10.0


def test_window_carries_the_evaluations_of_its_own_positions():
    game = replay()
    with pytest.raises(KeyError):
        game.window(14)  # not evaluated yet, so it must not be scored
    game.pawns = {position: position / 10 for position in range(1, 15)}
    window = game.window(14)
    assert window.side_to_move == "white" and len(window.evaluation) == len(window.white_clock) == 14
    assert window.evaluation[-1] == pytest.approx(1.4) and window.evaluation[0] == pytest.approx(0.1)


def test_replaying_recorded_feed_lines():
    game = replay()
    assert (game.ply, game.status, game.finished) == (32, "resign", True)
    assert (game.time_control, game.white, game.black) == ("600+0", {"rating": 2376, "title": "FM"}, {"rating": 2530, "title": None})
    assert (game.moves[1], game.white_clock[-1], game.black_clock[-1]) == ("d2d4", 561, 532)
    for line in RECORDED.splitlines():  # a reconnect replays the game from move one
        assert not game.feed(json.loads(line))
    game.pawns = dict.fromkeys(range(1, 40), 0.0)
    window = game.window(13)
    assert (window.side_to_move, window.ply, len(window.white_clock)) == ("black", 13, 13)
    assert window.white_clock[-1] == game.white_clock[13]  # White made ply 13
    assert window.black_clock[-1] == game.black_clock[12]
    shown = json.dumps(game.view()).lower()
    assert "manvith" not in shown and "twisha" not in shown
    with pytest.raises(ValueError, match="arrived after"):
        replay(10).feed(json.loads(RECORDED.splitlines()[20]))


def test_blunder_needs_the_movers_own_drop_of_three_tenths():
    game = replay()
    white, black = decide(game, 12), decide(game, 13)
    game.chances = dict.fromkeys(range(12, 19), 0.2)
    game.chances[15] = -0.15  # White's move at ply 15 loses 0.35; Black's reply at ply 16 wins it back for White
    assert game.settle() == [white, black]
    assert (white.blunder, white.worst_ply, white.worst_move) == (True, 15, game.moves[15])
    assert white.drop == pytest.approx(0.35)
    assert (black.blunder, black.worst_ply) == (True, 16)  # Black gave 0.35 back
    quiet = decide(game, 20)
    game.chances.update(dict.fromkeys(range(20, 26), 0.5) | {23: 0.21})  # 0.29 is not enough
    assert game.settle() == [quiet] and quiet.blunder is False


def test_decision_resolves_only_after_three_own_moves_are_evaluated():
    game = replay(18)  # description, starting position and 16 plies
    decision = decide(game, 12)
    game.chances = dict.fromkeys(range(12, 17), 0.0)
    assert game.ply == 16 and game.settle() == []  # the third move, ply 17, is not played yet
    game.feed(json.loads(RECORDED.splitlines()[18]))
    assert game.settle() == []  # played but not evaluated
    game.chances[17] = 0.0
    assert game.settle() == [decision] and decision.blunder is False and not game.open


def test_game_ending_early_voids_the_decision():
    game = replay()
    void, kept = decide(game, 28), decide(game, 27)
    game.chances = dict.fromkeys(range(27, 33), 0.0) | {30: 0.9}  # Black's ply 30 and White's ply 31 both blunder
    assert game.settle() == [void, kept]
    assert (void.void, void.blunder) == (True, None)
    assert (kept.void, kept.blunder) == (False, True)


def test_scoreboard_arithmetic():
    board = Scoreboard()
    for speed, called, blunder in [
        ("bullet", True, True),
        ("bullet", True, False),
        ("bullet", False, False),
        ("bullet", False, False),
        ("rapid", False, True),
        ("rapid", True, None),  # void: counted apart
    ]:
        board.add(speed, called=called, blunder=blunder)
    everything = board.snapshot()["all"]
    assert everything == {"resolved": 5, "blunders": 2, "calls": 2, "hits": 1, "void": 1, "base_rate": 0.4, "precision": 0.5, "lift": 1.25}
    assert board.snapshot()["bullet"]["lift"] == pytest.approx(0.5 / 0.25)
    assert board.snapshot()["rapid"] | {"base_rate": 1.0} == board.snapshot()["rapid"]
    assert board.snapshot()["rapid"]["precision"] is None and board.snapshot()["blitz"]["lift"] is None


def test_log_accumulates_across_restarts_per_model(tmp_path):
    log = tmp_path / "live-log.jsonl"
    other = {"model": "some other model", "threshold": 0.4, "t": "2026-11-20T10:00:00+00:00"}
    log.write_text(json.dumps(other | {"speed": "blitz", "called": True, "blunder": True}) + "\n")
    watcher = Watcher(DevHeuristicPredictor(), 0.4, log_path=log)
    assert watcher.board.snapshot()["all"]["resolved"] == 0
    game = replay()
    decide(game, 12)
    game.chances = dict.fromkeys(range(12, 18), 0.0) | {13: -0.5}
    watcher.settle(game)
    assert len(game.resolved_calls) == 1 and len(log.read_text().splitlines()) == 2
    again = Watcher(DevHeuristicPredictor(), 0.4, log_path=log).board.snapshot()
    assert (again["all"]["resolved"], again["rapid"]["hits"], again["rapid"]["precision"]) == (1, 1, 1.0)
    assert Watcher(DevHeuristicPredictor(), 0.2, log_path=log).board.snapshot()["all"]["resolved"] == 0  # another threshold


def test_dev_heuristic_rates_time_trouble_higher():
    predictor = DevHeuristicPredictor()
    game = replay()
    game.pawns = dict.fromkeys(range(1, 40), 0.0)
    calm = predictor.p_blunder(game.window(20))
    game.white_clock[15:] = [40 - 6 * i for i in range(len(game.white_clock) - 15)]  # White now burns 12 s a move
    assert 0 < calm < predictor.p_blunder(game.window(20)) < 1
