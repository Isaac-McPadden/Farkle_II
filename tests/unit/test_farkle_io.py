import csv
import multiprocessing as mp

from pytest import MonkeyPatch

import farkle.farkle_io as fio
from farkle.strategies import ThresholdStrategy


def test_writer_worker_appends(tmp_path: fio.Path):
    header = ["a", "b"]
    out = tmp_path / "out.csv"

    # first batch writes header
    q1 = mp.Queue()
    q1.put({"a": 1, "b": 2})
    q1.put(None)
    fio._writer_worker(q1, str(out), header)

    # second batch should append without another header
    q2 = mp.Queue()
    q2.put({"a": 3, "b": 4})
    q2.put(None)
    fio._writer_worker(q2, str(out), header)

    with out.open() as fh:
        rows = list(csv.DictReader(fh))

    assert rows == [
        {"a": "1", "b": "2"},
        {"a": "3", "b": "4"},
    ]


def test_single_game_row(monkeypatch: MonkeyPatch):
    expected = {
        "winner": "P2",
        "winning_score": 99,
        "n_rounds": 7,
        "P2_strategy": "S",
    }

    def fake_play(seed, strategies, target):  # noqa: ARG001
        return expected

    monkeypatch.setattr(fio, "_play_game", fake_play, raising=True)

    strat = [ThresholdStrategy(score_threshold=0, dice_threshold=6)]
    row = fio._single_game_row(5, 123, strat, 1000)

    assert row == {
        "game_id": 5,
        "winner": "P2",
        "winning_score": 99,
        "winner_strategy": "S",
        "n_rounds": 7,
    }


def test_single_game_row_mp(monkeypatch: MonkeyPatch):
    ret = {
        "winner": "P1",
        "winning_score": 42,
        "n_rounds": 3,
        "P1_strategy": "T",
    }

    monkeypatch.setattr(fio, "_play_game", lambda *a, **k: ret, raising=True)  # noqa: ARG005
    strat = [ThresholdStrategy(score_threshold=100, dice_threshold=2)]

    gid, row = fio._single_game_row_mp((2, 77, strat, 500))

    assert gid == 2
    assert row == {
        "game_id": 2,
        "winner": "P1",
        "winning_score": 42,
        "winner_strategy": "T",
        "n_rounds": 3,
    }
