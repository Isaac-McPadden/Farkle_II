from pathlib import Path
from typing import Any, Callable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import trueskill

import farkle.analysis.run_trueskill as rt
from farkle.analysis.trueskill_screening import (
    ScreeningRatingCell,
    publish_rating_cell_contract,
)
from farkle.config import AppConfig, IOConfig
from farkle.utils.artifact_contract import ArtifactContractError, sha256_file, sidecar_path
from farkle.utils.authenticated_contract import CodeIdentity


class _DummyRating:
    def __init__(self, mu: float = 0.0, sigma: float = 1.0) -> None:
        self.mu = mu
        self.sigma = sigma


class _DummyEnv:
    def __init__(self) -> None:
        self.rate_calls: list[list[int]] = []

    def create_rating(self, mu: float = 0.0, sigma: float = 1.0) -> _DummyRating:
        return _DummyRating(mu, sigma)

    def rate(self, teams: list[list[_DummyRating]], ranks: list[int]) -> list[list[_DummyRating]]:
        self.rate_calls.append(list(ranks))
        result: list[list[_DummyRating]] = []
        for team, rank in zip(teams, ranks, strict=True):
            base = team[0]
            result.append([_DummyRating(float(rank), base.sigma)])
        return result


@pytest.fixture()
def sample_parquet(tmp_path: Path) -> Path:
    table = pa.table(
        {
            "winner_seat": ["P1", "P2", "P1", "P2", "P1"],
            "P1_strategy": ["A", "B", "C", "D", "E"],
            "P2_strategy": ["B", "C", "D", "E", "F"],
        }
    )
    path = tmp_path / "games.parquet"
    pq.write_table(table, path, row_group_size=3)
    return path


def test_stream_batches_respects_resume_offsets(sample_parquet: Path) -> None:
    batches = list(
        rt._stream_batches(
            sample_parquet,
            ["winner_seat", "P1_strategy", "P2_strategy"],
            batch_rows=2,
        )
    )
    assert [(rg, bi, batch.num_rows) for rg, bi, batch in batches] == [
        (0, 0, 2),
        (0, 1, 1),
        (1, 0, 2),
    ]
    assert batches[0][2].column("P1_strategy").to_pylist() == ["A", "B"]

    resumed = list(
        rt._stream_batches(
            sample_parquet,
            ["winner_seat", "P1_strategy"],
            start_row_group=0,
            start_batch_idx=1,
            batch_rows=2,
        )
    )
    assert [(rg, bi) for rg, bi, _ in resumed] == [(0, 1), (1, 0)]
    assert resumed[0][2].column("P1_strategy").to_pylist() == ["C"]

    later_groups = list(
        rt._stream_batches(
            sample_parquet,
            ["winner_seat"],
            start_row_group=1,
            batch_rows=2,
        )
    )
    assert [(rg, bi, batch.num_rows) for rg, bi, batch in later_groups] == [(1, 0, 2)]


def test_players_and_ranks_use_only_completed_canonical_rows(tmp_path: Path) -> None:
    table = pa.table(
        {
            "termination_status": ["completed", "safety_limit", "completed"],
            "outcome_schema_version": [2, 2, 2],
            "winner_seat": ["P1", None, "P2"],
            "P1_strategy": ["alpha", "delta", "zeta"],
            "P2_strategy": ["beta", "epsilon", "eta"],
            "P3_strategy": ["gamma", "iota", "theta"],
            "P1_rank": pa.array([1, None, 3], type=pa.int64()),
            "P2_rank": pa.array([2, None, 1], type=pa.int64()),
            "P3_rank": pa.array([3, None, 2], type=pa.int64()),
        }
    )
    path = tmp_path / "precedence.parquet"
    pq.write_table(table, path, row_group_size=3)

    batch = next(
        rt._stream_batches(
            path,
            list(table.schema.names),
            batch_rows=10,
        )
    )[2]
    rows = list(rt._players_and_ranks_from_batch(batch, 3))
    assert rows == [
        (["alpha", "beta", "gamma"], [0, 1, 2]),
        (["zeta", "eta", "theta"], [2, 0, 1]),
    ]


def test_safety_limit_rows_cannot_carry_ranks_or_become_draws() -> None:
    table = pa.table(
        {
            "termination_status": ["safety_limit"],
            "outcome_schema_version": [2],
            "winner_seat": pa.array([None], type=pa.string()),
            "P1_strategy": ["A"],
            "P2_strategy": ["B"],
            "P1_rank": pa.array([1], type=pa.int64()),
            "P2_rank": pa.array([1], type=pa.int64()),
        }
    )
    with pytest.raises(ValueError, match="null winner and null ranks"):
        list(rt._players_and_ranks_from_batch(table, 2))


def test_rate_block_worker_resumes_from_checkpoint(tmp_path: Path) -> None:
    root = tmp_path / "analysis"
    data_dir = root / "by_k" / "2p"
    data_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = data_dir
    block_dir = tmp_path / "results" / "2_players"
    block_dir.mkdir(parents=True, exist_ok=True)
    np.save(block_dir / "keepers_2.npy", np.array(["A", "C"]))

    table = pa.table(
        {
            "termination_status": ["completed", "completed"],
            "outcome_schema_version": [2, 2],
            "winner_seat": ["P1", "P2"],
            "P1_strategy": ["A", "A"],
            "P2_strategy": ["B", "C"],
            "P1_rank": pa.array([1, 2], type=pa.int64()),
            "P2_rank": pa.array([2, 1], type=pa.int64()),
        }
    )
    row_file = data_dir / "2p_ingested_rows.parquet"
    pq.write_table(table, row_file, row_group_size=1)

    ratings_ck = checkpoint_dir / "ratings_2.checkpoint.parquet"
    rt._save_ratings_parquet(
        ratings_ck, {"A": trueskill.TrueSkill().create_rating(mu=25.0, sigma=8.0)}
    )
    ck_path = checkpoint_dir / "ratings_2.ckpt.json"
    rt._save_block_ckpt(
        ck_path,
        rt._BlockCkpt(
            row_file=str(row_file),
            row_group=1,
            batch_index=0,
            games_done=0,
            ratings_path=str(ratings_ck),
            freshness_sha256="a" * 64,
            attempted_games=1,
            completed_games=1,
            excluded_safety_limit_games=0,
            strategy_attempted_exposures={"A": 1, "C": 0},
            strategy_completed_exposures={"A": 1, "C": 0},
            strategy_excluded_safety_limit_exposures={"A": 0, "C": 0},
            strategy_performed_updates={"A": 0, "C": 0},
        ),
    )

    player_count, games = rt._rate_block_worker(
        str(block_dir),
        str(root),
        "",
        batch_rows=1,
        resume=True,
        checkpoint_every_batches=1,
        row_data_dir=str(root),
        curated_rows_name="2p_ingested_rows.parquet",
        cell_freshness_sha256="a" * 64,
    )
    assert player_count == "2"
    assert games == 1

    ratings = rt._load_ratings_parquet(data_dir / "ratings_2.parquet")
    assert set(ratings) == {"A", "C"}
    assert "B" not in ratings
    assert not (data_dir / "ratings_2.ckpt.json").exists()
    assert not (data_dir / "ratings_2.checkpoint.parquet").exists()


def _run_rating_fixture(
    tmp_path: Path,
    rows: list[dict[str, object]],
    *,
    keepers: tuple[str, ...],
) -> tuple[dict[str, rt.RatingStats], rt._ShardDoneStamp]:
    root = tmp_path / "analysis"
    block = tmp_path / "results" / "2_players"
    block.mkdir(parents=True)
    np.save(block / "keepers_2.npy", np.array(keepers))
    source = tmp_path / "curated" / "by_k" / "2p" / "games.parquet"
    source.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), source)
    _player_count, updates = rt._rate_block_worker(
        str(block),
        str(root),
        "_seed11",
        batch_rows=2,
        resume=False,
        checkpoint_every_batches=1,
        row_data_dir=str(tmp_path / "curated"),
        curated_rows_name="games.parquet",
        cell_freshness_sha256="a" * 64,
        root_seed=11,
    )
    rating_path = root / "by_k" / "2p" / "ratings_2_seed11.parquet"
    stamp = rt._load_done_stamp(root / "by_k" / "2p" / "ratings_2_seed11.done.json")
    assert stamp is not None
    assert updates == stamp.performed_update_games
    return rt._load_ratings_parquet(rating_path), stamp


def _completed_row(winner: str) -> dict[str, object]:
    return {
        "termination_status": "completed",
        "outcome_schema_version": 2,
        "winner_seat": winner,
        "P1_strategy": "A",
        "P2_strategy": "B",
        "P1_rank": 1 if winner == "P1" else 2,
        "P2_rank": 2 if winner == "P1" else 1,
    }


def test_all_completed_ratings_are_unchanged(tmp_path: Path) -> None:
    ratings, stamp = _run_rating_fixture(
        tmp_path,
        [_completed_row("P1"), _completed_row("P2")],
        keepers=("A", "B"),
    )
    env = trueskill.TrueSkill()
    expected_a = env.create_rating()
    expected_b = env.create_rating()
    for ranks in ([0, 1], [1, 0]):
        updated = env.rate([[expected_a], [expected_b]], ranks=ranks)
        expected_a, expected_b = updated[0][0], updated[1][0]

    assert ratings["A"].mu == pytest.approx(expected_a.mu)
    assert ratings["A"].sigma == pytest.approx(expected_a.sigma)
    assert ratings["B"].mu == pytest.approx(expected_b.mu)
    assert ratings["B"].sigma == pytest.approx(expected_b.sigma)
    assert stamp.attempted_games == stamp.completed_games == 2
    assert stamp.excluded_safety_limit_games == 0
    assert stamp.performed_update_games == 2


def test_mixed_support_excludes_safety_and_retains_prior_only_strategy(
    tmp_path: Path,
) -> None:
    safety_ac = {
        "termination_status": "safety_limit",
        "outcome_schema_version": 2,
        "winner_seat": None,
        "P1_strategy": "A",
        "P2_strategy": "C",
        "P1_rank": None,
        "P2_rank": None,
    }
    safety_cb = {
        **safety_ac,
        "P1_strategy": "C",
        "P2_strategy": "B",
    }
    ratings, stamp = _run_rating_fixture(
        tmp_path,
        [_completed_row("P1"), safety_ac, safety_cb, _completed_row("P2")],
        keepers=("A", "B", "C"),
    )

    assert (
        stamp.attempted_games,
        stamp.completed_games,
        stamp.excluded_safety_limit_games,
        stamp.performed_update_games,
    ) == (4, 2, 2, 2)
    assert ratings["C"].strategy_attempted_exposures == 2
    assert ratings["C"].strategy_completed_exposures == 0
    assert ratings["C"].strategy_excluded_safety_limit_exposures == 2
    assert ratings["C"].strategy_performed_updates == 0
    assert ratings["C"].rating_status == "prior_only_unrated"
    assert ratings["C"].mu == pytest.approx(trueskill.Rating().mu)
    assert ratings["C"].sigma == pytest.approx(trueskill.Rating().sigma)
    for stats in ratings.values():
        assert stats.strategy_attempted_exposures == (
            stats.strategy_completed_exposures + stats.strategy_excluded_safety_limit_exposures
        )


@pytest.mark.parametrize(
    ("loader", "filename"),
    [
        (rt._load_block_ckpt, "block.ckpt.json"),
    ],
)
def test_load_ckpts_handle_missing_and_invalid(
    tmp_path: Path, loader: Callable[[Path], Any], filename: str
) -> None:
    path = tmp_path / filename
    assert loader(path) is None

    path.write_text("{not json")
    assert loader(path) is None


def test_rate_block_worker_rejects_missing_canonical_coordinates(tmp_path: Path) -> None:
    root = tmp_path / "analysis"
    root.mkdir(parents=True, exist_ok=True)
    block = tmp_path / "results" / "2_players"
    block.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError, match="explicit canonical curated-row"):
        rt._rate_block_worker(
            str(block),
            str(root),
            "",
            batch_rows=1,
            resume=False,
            checkpoint_every_batches=1,
            cell_freshness_sha256="a" * 64,
        )


@pytest.mark.parametrize(
    "mutation",
    ["unchanged", "force", "input", "parameter", "output", "code", "method", "sidecar"],
)
def test_trueskill_cell_authenticated_reuse_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    root = tmp_path / "analysis"
    block = tmp_path / "results" / "2_players"
    block.mkdir(parents=True)
    np.save(block / "keepers_2.npy", np.array(["A", "B"]))
    source = tmp_path / "curated" / "by_k" / "2p" / "game_rows.parquet"
    source.parent.mkdir(parents=True)
    games = pa.table(
        {
            "termination_status": ["completed", "completed"],
            "outcome_schema_version": [2, 2],
            "winner_seat": ["P1", "P2"],
            "P1_strategy": ["A", "A"],
            "P2_strategy": ["B", "B"],
            "P1_rank": [1, 2],
            "P2_rank": [2, 1],
        }
    )
    pq.write_table(games, source)
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "cfg"))
    cfg._code_identity = CodeIdentity(
        commit="a" * 40,
        policy="development_dirty",
        state="development_dirty",
        dirty_fingerprint_sha256="b" * 64,
    )
    freshness = "c" * 64
    rt._rate_block_worker(
        str(block),
        str(root),
        "_seed11",
        batch_rows=10,
        resume=True,
        row_data_dir=str(tmp_path / "curated"),
        curated_rows_name="game_rows.parquet",
        cell_freshness_sha256=freshness,
        root_seed=11,
    )
    rating = root / "by_k" / "2p" / "ratings_2_seed11.parquet"
    done = root / "by_k" / "2p" / "ratings_2_seed11.done.json"
    stamp = rt._load_done_stamp(done)
    assert stamp is not None
    cell = ScreeningRatingCell(11, 2, rating, source)
    rt._seal_rating_cell_completion(
        cfg,
        cell=cell,
        done_path=done,
        stamp=stamp,
        source_path=source,
        freshness=freshness,
    )
    assert not rt._done_stamp_matches(
        rt._load_done_stamp(done),
        parquet_path=rating,
        source_path=source,
        freshness=freshness,
        root_seed=12,
        player_count=2,
    )

    if mutation == "input":
        pq.write_table(pa.concat_tables([games, games.slice(0, 1)]), source)
    elif mutation in {"parameter", "code"}:
        freshness = "d" * 64
    elif mutation == "output":
        with rating.open("ab") as handle:
            handle.write(b"changed")
    elif mutation == "method":
        monkeypatch.setattr(
            rt,
            "TRUESKILL_CELL_METHOD_VERSION",
            rt.TRUESKILL_CELL_METHOD_VERSION + 1,
        )
    elif mutation == "sidecar":
        sidecar_path(rating).write_text("{}", encoding="utf-8")

    writes = 0
    original_save = rt._save_ratings_parquet

    def counted_save(path: Path, ratings: object) -> None:
        nonlocal writes
        writes += 1
        original_save(path, ratings)  # type: ignore[arg-type]

    monkeypatch.setattr(rt, "_save_ratings_parquet", counted_save)
    rt._rate_block_worker(
        str(block),
        str(root),
        "_seed11",
        batch_rows=10,
        resume=mutation != "force",
        row_data_dir=str(tmp_path / "curated"),
        curated_rows_name="game_rows.parquet",
        cell_freshness_sha256=freshness,
        root_seed=11,
    )
    assert writes == 0 if mutation == "unchanged" else writes > 0


def test_trueskill_corruption_cannot_be_blessed_and_missing_sidecar_recovery_is_bound(
    tmp_path: Path,
) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "cfg"))
    cfg._code_identity = CodeIdentity(
        commit="a" * 40,
        policy="development_dirty",
        state="development_dirty",
        dirty_fingerprint_sha256="b" * 64,
    )
    source = tmp_path / "source.parquet"
    rating = tmp_path / "rating.parquet"
    pq.write_table(pa.table({"winner_seat": ["P1"]}), source)
    support = {
        "A": rt.RatingStats(
            25.0,
            8.0,
            1,
            1,
            0,
            1,
            "evidence_backed_completed_games",
            1,
            1,
            0,
            1,
        ),
        "B": rt.RatingStats(
            25.0,
            8.0,
            1,
            1,
            0,
            1,
            "evidence_backed_completed_games",
            1,
            1,
            0,
            1,
        ),
    }
    pq.write_table(rt._ratings_to_table(support), rating)
    cell = ScreeningRatingCell(11, 2, rating, source)
    stamp = rt._ShardDoneStamp(
        shard_key="k=2",
        parquet_path=str(rating),
        rows=1,
        created_at=1.0,
        root_seed=11,
        player_count=2,
        method_version=rt.TRUESKILL_CELL_METHOD_VERSION,
        source_sha256=sha256_file(source),
        source_sidecar_sha256=None,
        parquet_sha256=sha256_file(rating),
        freshness_sha256="c" * 64,
        sidecar_sha256=None,
        attempted_games=1,
        completed_games=1,
        excluded_safety_limit_games=0,
        performed_update_games=1,
    )
    done = tmp_path / "rating.done.json"
    rt._save_done_stamp(done, stamp)
    sealed = rt._seal_rating_cell_completion(
        cfg,
        cell=cell,
        done_path=done,
        stamp=stamp,
        source_path=source,
        freshness="c" * 64,
    )
    expected_sidecar = sealed.sidecar_sha256
    sidecar_path(rating).unlink()
    recovered = rt._seal_rating_cell_completion(
        cfg,
        cell=cell,
        done_path=done,
        stamp=sealed,
        source_path=source,
        freshness="c" * 64,
    )
    assert recovered.sidecar_sha256 == expected_sidecar

    support["A"].mu = 99.0
    support["A"].sigma = 1.0
    pq.write_table(rt._ratings_to_table(support), rating)
    sidecar_path(rating).unlink(missing_ok=True)
    publish_rating_cell_contract(
        cfg,
        cell,
        completed_artifact_sha256=sha256_file(rating),
        code_revision=rt._trueskill_code_revision(cfg),
    )
    with pytest.raises(ArtifactContractError, match="does not bind current bytes"):
        rt._seal_rating_cell_completion(
            cfg,
            cell=cell,
            done_path=done,
            stamp=sealed,
            source_path=source,
            freshness="c" * 64,
        )


def test_trueskill_cell_freshness_binds_parameter_code_and_method(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path))
    cfg._code_identity = CodeIdentity(
        commit="a" * 40,
        policy="development_dirty",
        state="development_dirty",
        dirty_fingerprint_sha256="b" * 64,
    )
    baseline = rt._trueskill_cell_freshness(cfg)

    cfg.trueskill.beta += 1.0
    parameter_changed = rt._trueskill_cell_freshness(cfg)
    assert parameter_changed != baseline

    cfg._code_identity = CodeIdentity(
        commit="c" * 40,
        policy="development_dirty",
        state="development_dirty",
        dirty_fingerprint_sha256="d" * 64,
    )
    code_changed = rt._trueskill_cell_freshness(cfg)
    assert code_changed != parameter_changed

    monkeypatch.setattr(
        rt,
        "TRUESKILL_CELL_METHOD_VERSION",
        rt.TRUESKILL_CELL_METHOD_VERSION + 1,
    )
    assert rt._trueskill_cell_freshness(cfg) != code_changed
