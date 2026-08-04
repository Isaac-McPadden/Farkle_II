from pathlib import Path
from typing import Any, Callable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import trueskill
from tests.helpers.artifact_sidecars import (
    clean_test_code_identity,
    make_authenticated_v3_config,
    mutate_artifact_bytes,
    mutate_json_identity_leaf,
    publish_v3_parquet,
)

import farkle.analysis.run_trueskill as rt
from farkle.analysis.trueskill_screening import (
    ScreeningRatingCell,
)
from farkle.utils.artifact_contract import ArtifactContractError, sha256_file, sidecar_path


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
            "P1_strategy": ["1", "2", "3", "4", "5"],
            "P2_strategy": ["2", "3", "4", "5", "6"],
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
    assert batches[0][2].column("P1_strategy").to_pylist() == ["1", "2"]

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
    assert resumed[0][2].column("P1_strategy").to_pylist() == ["3"]

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
            "P1_strategy": pa.array([11, 14, 16], type=pa.int32()),
            "P2_strategy": pa.array([12, 15, 17], type=pa.int32()),
            "P3_strategy": pa.array([13, 19, 18], type=pa.int32()),
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
        (["11", "12", "13"], [0, 1, 2]),
        (["16", "17", "18"], [2, 0, 1]),
    ]


def test_safety_limit_rows_cannot_carry_ranks_or_become_draws() -> None:
    table = pa.table(
        {
            "termination_status": ["safety_limit"],
            "outcome_schema_version": [2],
            "winner_seat": pa.array([None], type=pa.string()),
            "P1_strategy": pa.array([1], type=pa.int32()),
            "P2_strategy": pa.array([2], type=pa.int32()),
            "P1_rank": pa.array([1], type=pa.int64()),
            "P2_rank": pa.array([1], type=pa.int64()),
        }
    )
    with pytest.raises(ValueError, match="null winner and null ranks"):
        list(rt._players_and_ranks_from_batch(table, 2))


def test_rate_block_worker_resumes_from_checkpoint(tmp_path: Path) -> None:
    cfg = make_authenticated_v3_config(tmp_path, name="resume", root_seed=11)
    root = cfg.trueskill_stage_dir
    data_dir = cfg.trueskill_stage_dir / "by_k" / "2p"
    data_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = data_dir
    block_dir = cfg.n_dir(2)
    block_dir.mkdir(parents=True, exist_ok=True)
    np.save(block_dir / "keepers_2.npy", np.array([1, 3], dtype=np.int32))

    table = pa.table(
        {
            "termination_status": ["completed", "completed"],
            "outcome_schema_version": [2, 2],
            "winner_seat": ["P1", "P2"],
            "P1_strategy": pa.array([1, 1], type=pa.int32()),
            "P2_strategy": pa.array([2, 3], type=pa.int32()),
            "P1_rank": pa.array([1, 2], type=pa.int64()),
            "P2_rank": pa.array([2, 1], type=pa.int64()),
        }
    )
    row_file = cfg.ingested_rows_curated(2)
    publish_v3_parquet(
        cfg,
        row_file,
        table,
        stage_key="curate",
        producer="curate",
        operation="curate_game_rows",
    )

    ratings_ck = checkpoint_dir / "ratings_2.checkpoint.parquet"
    rt._save_ratings_parquet(
        ratings_ck, {"1": trueskill.TrueSkill().create_rating(mu=25.0, sigma=8.0)}
    )
    ck_path = checkpoint_dir / "ratings_2.ckpt.json"
    rt._save_block_ckpt(
        ck_path,
        rt._BlockCkpt(
            row_file=str(row_file),
            row_group=0,
            batch_index=1,
            games_done=0,
            ratings_path=str(ratings_ck),
            freshness_sha256="a" * 64,
            attempted_games=1,
            completed_games=1,
            excluded_safety_limit_games=0,
            strategy_attempted_exposures={"1": 1, "3": 0},
            strategy_completed_exposures={"1": 1, "3": 0},
            strategy_excluded_safety_limit_exposures={"1": 0, "3": 0},
            strategy_performed_updates={"1": 0, "3": 0},
        ),
    )

    player_count, games = rt._rate_block_worker(
        str(block_dir),
        str(root),
        "",
        batch_rows=1,
        resume=True,
        checkpoint_every_batches=1,
        row_data_dir=str(cfg.curate_stage_dir),
        curated_rows_name=cfg.curated_rows_name,
        cell_freshness_sha256="a" * 64,
        root_seed=11,
    )
    assert player_count == "2"
    assert games == 1

    ratings = rt._load_ratings_parquet(data_dir / "ratings_2.parquet")
    assert set(ratings) == {"1", "3"}
    assert "2" not in ratings
    assert not (data_dir / "ratings_2.ckpt.json").exists()
    assert not (data_dir / "ratings_2.checkpoint.parquet").exists()


def _run_rating_fixture(
    tmp_path: Path,
    rows: list[dict[str, object]],
    *,
    keepers: tuple[int, ...],
) -> tuple[dict[str, rt.RatingStats], rt._ShardDoneStamp]:
    cfg = make_authenticated_v3_config(tmp_path, name="rating", root_seed=11)
    root = cfg.trueskill_stage_dir
    block = cfg.n_dir(2)
    block.mkdir(parents=True)
    np.save(block / "keepers_2.npy", np.array(keepers, dtype=np.int32))
    source = cfg.ingested_rows_curated(2)
    publish_v3_parquet(
        cfg,
        source,
        pa.Table.from_pylist(rows),
        stage_key="curate",
        producer="curate",
        operation="curate_game_rows",
    )
    _player_count, updates = rt._rate_block_worker(
        str(block),
        str(root),
        "_seed11",
        batch_rows=2,
        resume=False,
        checkpoint_every_batches=1,
        row_data_dir=str(cfg.curate_stage_dir),
        curated_rows_name=cfg.curated_rows_name,
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
        "P1_strategy": 1,
        "P2_strategy": 2,
        "P1_rank": 1 if winner == "P1" else 2,
        "P2_rank": 2 if winner == "P1" else 1,
    }


def test_all_completed_ratings_are_unchanged(tmp_path: Path) -> None:
    ratings, stamp = _run_rating_fixture(
        tmp_path,
        [_completed_row("P1"), _completed_row("P2")],
        keepers=(1, 2),
    )
    env = trueskill.TrueSkill()
    expected_a = env.create_rating()
    expected_b = env.create_rating()
    for ranks in ([0, 1], [1, 0]):
        updated = env.rate([[expected_a], [expected_b]], ranks=ranks)
        expected_a, expected_b = updated[0][0], updated[1][0]

    assert ratings["1"].mu == pytest.approx(expected_a.mu)
    assert ratings["1"].sigma == pytest.approx(expected_a.sigma)
    assert ratings["2"].mu == pytest.approx(expected_b.mu)
    assert ratings["2"].sigma == pytest.approx(expected_b.sigma)
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
        "P1_strategy": 1,
        "P2_strategy": 3,
        "P1_rank": None,
        "P2_rank": None,
    }
    safety_cb = {
        **safety_ac,
        "P1_strategy": 3,
        "P2_strategy": 2,
    }
    ratings, stamp = _run_rating_fixture(
        tmp_path,
        [_completed_row("P1"), safety_ac, safety_cb, _completed_row("P2")],
        keepers=(1, 2, 3),
    )

    assert (
        stamp.attempted_games,
        stamp.completed_games,
        stamp.excluded_safety_limit_games,
        stamp.performed_update_games,
    ) == (4, 2, 2, 2)
    assert ratings["3"].strategy_attempted_exposures == 2
    assert ratings["3"].strategy_completed_exposures == 0
    assert ratings["3"].strategy_excluded_safety_limit_exposures == 2
    assert ratings["3"].strategy_performed_updates == 0
    assert ratings["3"].rating_status == "prior_only_unrated"
    assert ratings["3"].mu == pytest.approx(trueskill.Rating().mu)
    assert ratings["3"].sigma == pytest.approx(trueskill.Rating().sigma)
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
    [
        "unchanged",
        "force",
        "source",
        "schema",
        "conditioning",
        "parameter",
        "output",
        "code",
        "method",
        "sidecar",
        "completion",
    ],
)
def test_trueskill_cell_authenticated_reuse_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    cfg = make_authenticated_v3_config(tmp_path, name="trueskill", root_seed=11)
    root = cfg.trueskill_stage_dir
    block = cfg.n_dir(2)
    block.mkdir(parents=True)
    np.save(block / "keepers_2.npy", np.array([1, 2], dtype=np.int32))
    source = cfg.ingested_rows_curated(2)
    games = pa.table(
        {
            "termination_status": ["completed", "completed"],
            "outcome_schema_version": [2, 2],
            "winner_seat": ["P1", "P2"],
            "P1_strategy": pa.array([1, 1], type=pa.int32()),
            "P2_strategy": pa.array([2, 2], type=pa.int32()),
            "P1_rank": [1, 2],
            "P2_rank": [2, 1],
        }
    )
    publish_v3_parquet(
        cfg,
        source,
        games,
        stage_key="curate",
        producer="curate",
        operation="curate_game_rows",
    )
    freshness = "c" * 64
    rt._rate_block_worker(
        str(block),
        str(root),
        "_seed11",
        batch_rows=10,
        resume=True,
        row_data_dir=str(cfg.curate_stage_dir),
        curated_rows_name=cfg.curated_rows_name,
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
    assert rt._done_stamp_matches(
        rt._load_done_stamp(done),
        parquet_path=rating,
        source_path=source,
        freshness=freshness,
        root_seed=11,
        player_count=2,
    ), "valid authenticated TrueSkill control must be accepted before mutation"
    assert not rt._done_stamp_matches(
        rt._load_done_stamp(done),
        parquet_path=rating,
        source_path=source,
        freshness=freshness,
        root_seed=12,
        player_count=2,
    )

    if mutation == "source":
        publish_v3_parquet(
            cfg,
            source,
            pa.concat_tables([games, games.slice(0, 1)]),
            stage_key="curate",
            producer="curate",
            operation="curate_game_rows",
        )
    elif mutation == "schema":
        publish_v3_parquet(
            cfg,
            source,
            games.append_column(
                "fixture_schema_marker",
                pa.array([1, 1], type=pa.int8()),
            ),
            stage_key="curate",
            producer="curate",
            operation="curate_game_rows",
        )
    elif mutation == "conditioning":
        publish_v3_parquet(
            cfg,
            source,
            games,
            stage_key="curate",
            producer="curate",
            operation="curate_game_rows",
            conditioning="termination_status == completed",
        )
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
    elif mutation == "completion":
        mutate_json_identity_leaf(
            done,
            ("freshness_sha256",),
            "e" * 64,
        )

    if mutation not in {"unchanged", "force"}:
        assert not rt._done_stamp_matches(
            rt._load_done_stamp(done),
            parquet_path=rating,
            source_path=source,
            freshness=freshness,
            root_seed=11,
            player_count=2,
        ), f"{mutation} must invalidate TrueSkill cell freshness before replay"

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
        row_data_dir=str(cfg.curate_stage_dir),
        curated_rows_name=cfg.curated_rows_name,
        cell_freshness_sha256=freshness,
        root_seed=11,
    )
    assert writes == 0 if mutation == "unchanged" else writes > 0


@pytest.mark.parametrize("mutation", ["missing_sidecar", "output_bytes", "completion_identity"])
def test_trueskill_corruption_cannot_be_blessed(
    tmp_path: Path,
    mutation: str,
) -> None:
    cfg = make_authenticated_v3_config(tmp_path, name="corruption", root_seed=11)
    source = publish_v3_parquet(
        cfg,
        cfg.ingested_rows_curated(2),
        pa.table(
            {
                "termination_status": ["completed"],
                "outcome_schema_version": [2],
                "winner_seat": ["P1"],
                "P1_strategy": pa.array([1], type=pa.int32()),
                "P2_strategy": pa.array([2], type=pa.int32()),
                "P1_rank": [1],
                "P2_rank": [2],
            }
        ),
        stage_key="curate",
        producer="curate",
        operation="curate_game_rows",
    )
    rating = cfg.trueskill_rating_path(2, root_seed=11)
    rating.parent.mkdir(parents=True, exist_ok=True)
    support = {
        "1": rt.RatingStats(
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
        "2": rt.RatingStats(
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
        source_sidecar_sha256=sha256_file(sidecar_path(source)),
        parquet_sha256=sha256_file(rating),
        freshness_sha256="c" * 64,
        sidecar_sha256=None,
        attempted_games=1,
        completed_games=1,
        excluded_safety_limit_games=0,
        performed_update_games=1,
    )
    done = rating.with_suffix(".done.json")
    rt._save_done_stamp(done, stamp)
    sealed = rt._seal_rating_cell_completion(
        cfg,
        cell=cell,
        done_path=done,
        stamp=stamp,
        source_path=source,
        freshness="c" * 64,
    )
    assert rt._done_stamp_matches(
        sealed,
        parquet_path=rating,
        source_path=source,
        freshness="c" * 64,
        root_seed=11,
        player_count=2,
    ), "valid authenticated TrueSkill control must be accepted before mutation"

    if mutation == "missing_sidecar":
        sidecar_path(rating).unlink()
        expected = "cannot reconstruct a missing sidecar"
    elif mutation == "output_bytes":
        mutate_artifact_bytes(rating)
        expected = "does not bind current bytes"
    else:
        mutate_json_identity_leaf(done, ("parquet_sha256",), "f" * 64)
        sealed = rt._load_done_stamp(done)
        assert sealed is not None
        expected = "does not bind current bytes"

    with pytest.raises(ArtifactContractError, match=expected):
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
    cfg = make_authenticated_v3_config(tmp_path, name="freshness")
    baseline = rt._trueskill_cell_freshness(cfg)

    cfg.trueskill.beta += 1.0
    parameter_changed = rt._trueskill_cell_freshness(cfg)
    assert parameter_changed != baseline

    cfg._code_identity = clean_test_code_identity("c" * 40)
    code_changed = rt._trueskill_cell_freshness(cfg)
    assert code_changed != parameter_changed

    monkeypatch.setattr(
        rt,
        "TRUESKILL_CELL_METHOD_VERSION",
        rt.TRUESKILL_CELL_METHOD_VERSION + 1,
    )
    assert rt._trueskill_cell_freshness(cfg) != code_changed
