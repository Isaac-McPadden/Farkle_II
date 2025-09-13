import pandas as pd
import pytest
from pytest import MonkeyPatch

import farkle.analysis.run_full_field as rf


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
        calls.append((kwargs["row_output_directory"], kwargs.get("num_shuffles")))

    monkeypatch.setattr("farkle.simulation.run_tournament.run_tournament", fake_run_tournament)
    monkeypatch.setattr(rf.mp, "set_start_method", lambda *a, **k: None)  # noqa: ARG005
    monkeypatch.chdir(tmp_path)

    rf.main()

    players = [2, 3, 4, 5, 6, 8, 10, 12]
    seen = sorted(int(p[0].parent.name.split("_")[0]) for p in calls)
    assert seen == players
    assert len(calls) == len(players)
    assert all(isinstance(nshuf, int) for _, nshuf in calls)


def test_main_skips_complete_and_resets_partial(
    monkeypatch: MonkeyPatch, tmp_path: rf.Path
) -> None:
    calls = []

    def fake_run_tournament(**kwargs):
        row_dir = kwargs["row_output_directory"]
        # ensure partial directories were removed before calling
        assert not row_dir.exists()
        calls.append(kwargs["n_players"])

    monkeypatch.setattr("farkle.simulation.run_tournament.run_tournament", fake_run_tournament)
    monkeypatch.setattr(rf.mp, "set_start_method", lambda *a, **k: None)  # noqa: ARG005
    monkeypatch.chdir(tmp_path)

    base = tmp_path / "data" / "results_seed_0"
    base.mkdir(parents=True)

    # create completed results for 2 players
    done = base / "2_players"
    done.mkdir()
    (done / "2p_checkpoint.pkl").write_text("done")
    (done / "2p_rows.parquet").write_text("rows")

    # create partial results for 3 players
    partial = base / "3_players"
    row_dir = partial / "3p_rows"
    row_dir.mkdir(parents=True)
    (row_dir / "a.parquet").write_text("partial")

    rf.main()

    # 2-player run skipped, 3-player run executed with reset
    players_called = calls
    assert 2 not in players_called
    assert 3 in players_called


def test_combo_complete_force_clean(tmp_path: rf.Path, caplog: pytest.LogCaptureFixture) -> None:
    n = 4
    out_dir = tmp_path
    row_dir = out_dir / f"{n}p_rows"
    row_dir.mkdir()
    (out_dir / f"{n}p_rows.parquet").write_text("rows")
    (out_dir / f"{n}p_checkpoint.pkl").write_text("done")

    with caplog.at_level("WARNING"):
        rf._combo_complete(out_dir, n)
    assert row_dir.exists()
    assert "--force-clean" in caplog.text

    caplog.clear()
    with caplog.at_level("WARNING"):
        rf._combo_complete(out_dir, n, force_clean=True)
    assert not row_dir.exists()
    assert "Deleting existing row directory" in caplog.text
