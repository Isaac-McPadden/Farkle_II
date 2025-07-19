import pandas as pd
from pytest import MonkeyPatch

import farkle.run_full_field as rf


def test_concat_row_shards(tmp_path: rf.Path):
    out_dir = tmp_path
    n = 3
    row_dir = out_dir / f"{n}p_rows"
    row_dir.mkdir()

    df1 = pd.DataFrame({"v": [1]})
    df2 = pd.DataFrame({"v": [2]})
    df1.to_parquet(row_dir / "a.parquet")
    df2.to_parquet(row_dir / "b.parquet")

    rf._concat_row_shards(out_dir, n)

    merged = out_dir / f"{n}p_rows.parquet"
    assert merged.exists()
    assert not row_dir.exists()
    assert list(pd.read_parquet(merged)["v"]) == [1, 2]


def test_main_invokes_run_tournament(monkeypatch: MonkeyPatch, tmp_path: rf.Path):
    calls = []

    def fake_run_tournament(**kwargs):
        calls.append(kwargs["row_output_directory"])

    monkeypatch.setattr("farkle.run_tournament.run_tournament", fake_run_tournament)
    monkeypatch.setattr(rf.importlib, "reload", lambda mod: mod)
    monkeypatch.setattr(rf.mp, "set_start_method", lambda *a, **k: None)  # noqa: ARG005
    monkeypatch.chdir(tmp_path)

    rf.main()

    players = [2, 3, 4, 5, 6, 8, 10, 12]
    seen = sorted(int(p.parent.name.split("_")[0]) for p in calls)
    assert seen == players
    assert len(calls) == len(players)
