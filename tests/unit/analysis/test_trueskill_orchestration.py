from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from farkle.analysis import run_trueskill as run_trueskill_module
from farkle.analysis import trueskill as trueskill_stage
from farkle.analysis.stage_registry import resolve_root_pair_stage_layout, resolve_stage_layout
from farkle.analysis.trueskill_screening import ScreeningRatingCell, publish_rating_cell_contract
from farkle.config import AppConfig, IOConfig, SimConfig
from farkle.orchestration.run_contexts import RootPairRunContext, SeedRunContext
from farkle.utils.artifact_contract import (
    ArtifactContractError,
    sha256_file,
    validate_artifact_sidecar,
)


def _root_context(tmp_path: Path, seed: int, ks: tuple[int, ...] = (2, 4)) -> SeedRunContext:
    cfg = AppConfig(
        io=IOConfig(results_dir_prefix=tmp_path / f"root_{seed}"),
        sim=SimConfig(seed=seed, seed_list=[seed], n_players_list=list(ks)),
    )
    cfg.set_stage_layout(resolve_stage_layout(cfg))
    return SeedRunContext.from_config(cfg)


def _write_rating_cell(context: SeedRunContext, k: int, *, valid_sidecar: bool = True) -> Path:
    path = context.config.trueskill_rating_path(k, root_seed=context.seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "strategy": ["A", "B"],
                "mu": [30.0 + context.seed / 100 + k / 1000, 20.0],
                "sigma": [2.0, 3.0],
            }
        ),
        path,
    )
    if valid_sidecar:
        publish_rating_cell_contract(
            context.config,
            ScreeningRatingCell(root_seed=context.seed, k=k, ratings_path=path),
            completed_artifact_sha256=sha256_file(path),
        )
    return path


def _pair_context(tmp_path: Path) -> RootPairRunContext:
    first = _root_context(tmp_path, 11)
    second = _root_context(tmp_path, 22)
    return RootPairRunContext.from_root_contexts(
        (first, second),
        pair_root=tmp_path / "pair",
    )


def test_root_pair_trueskill_aggregates_complete_root_k_cells(tmp_path: Path) -> None:
    context = _pair_context(tmp_path)
    sources = [
        _write_rating_cell(root_context, k)
        for root_context in context.root_contexts
        for k in (2, 4)
    ]

    trueskill_stage.run_root_pair(context.config, context.root_contexts)

    output = context.config.trueskill_candidate_contribution_path()
    frame = pq.read_table(output).to_pandas().set_index("strategy")
    assert frame.loc["A", "rating_cells_present"] == 4
    assert frame.loc["A", "rating_cells_required"] == 4
    assert frame.loc["A", "complete_support"]
    sidecar = validate_artifact_sidecar(
        output,
        expected={
            "scope": "across_k",
            "operation": "equal_root_k_percentile_mean",
            "seed_scope": "both_roots_combined",
        },
    )
    assert set(sidecar.source_artifacts) == {str(path) for path in sources}


def test_root_pair_trueskill_rejects_missing_and_invalid_cells(tmp_path: Path) -> None:
    context = _pair_context(tmp_path)
    for root_context in context.root_contexts:
        for k in (2, 4):
            if (root_context.seed, k) != (22, 4):
                _write_rating_cell(root_context, k)

    with pytest.raises(FileNotFoundError, match="rating cells are missing"):
        trueskill_stage.run_root_pair(context.config, context.root_contexts)

    _write_rating_cell(context.root_contexts[1], 4, valid_sidecar=False)
    with pytest.raises(ArtifactContractError):
        trueskill_stage.run_root_pair(context.config, context.root_contexts)


def test_root_pair_trueskill_rejects_duplicate_roots(tmp_path: Path) -> None:
    root = _root_context(tmp_path, 11)
    cfg = AppConfig(
        io=IOConfig(results_dir_prefix=tmp_path / "pair_results"),
        sim=SimConfig(seed=11, seed_list=[11, 11], n_players_list=[2]),
    )
    cfg.set_stage_layout(resolve_root_pair_stage_layout(cfg))

    with pytest.raises(ValueError, match="two distinct configured roots"):
        trueskill_stage.run_root_pair(cfg, (root, root))


def test_run_trueskill_root_rejects_incomplete_configured_k_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _root_context(tmp_path, 11)
    rating = _write_rating_cell(context, 2)
    monkeypatch.setattr(run_trueskill_module, "run_trueskill", lambda **_kwargs: None)
    monkeypatch.setattr(run_trueskill_module, "_resolve_root_row_data_dir", lambda _cfg: tmp_path)
    monkeypatch.setattr(
        run_trueskill_module,
        "_iter_rating_parquets",
        lambda _root, _suffix: [rating],
    )
    monkeypatch.setattr(
        run_trueskill_module,
        "_load_done_stamp",
        lambda _path: SimpleNamespace(parquet_path=str(rating), sidecar_sha256=None),
    )
    monkeypatch.setattr(run_trueskill_module, "_done_stamp_matches", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        run_trueskill_module,
        "_seal_rating_cell_completion",
        lambda _cfg, **kwargs: kwargs["stamp"],
    )

    with pytest.raises(RuntimeError, match="exactly every configured root/k cell"):
        run_trueskill_module.run_trueskill_root(context.config)


def test_run_trueskill_root_rejects_extra_k_cells(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _root_context(tmp_path, 11, ks=(2,))
    ratings = [_write_rating_cell(context, k) for k in (2, 4)]
    by_done_name = {
        f"ratings_{k}_seed11.done.json": rating for k, rating in zip((2, 4), ratings, strict=True)
    }
    monkeypatch.setattr(run_trueskill_module, "run_trueskill", lambda **_kwargs: None)
    monkeypatch.setattr(run_trueskill_module, "_resolve_root_row_data_dir", lambda _cfg: tmp_path)
    monkeypatch.setattr(
        run_trueskill_module,
        "_iter_rating_parquets",
        lambda _root, _suffix: ratings,
    )
    monkeypatch.setattr(
        run_trueskill_module,
        "_load_done_stamp",
        lambda path: SimpleNamespace(
            parquet_path=str(by_done_name[path.name]), sidecar_sha256=None
        ),
    )
    monkeypatch.setattr(run_trueskill_module, "_done_stamp_matches", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        run_trueskill_module,
        "_seal_rating_cell_completion",
        lambda _cfg, **kwargs: kwargs["stamp"],
    )

    with pytest.raises(RuntimeError, match=r"extra=\[\(11, 4\)\]"):
        run_trueskill_module.run_trueskill_root(context.config)


def test_parallel_trueskill_block_failure_propagates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataroot = tmp_path / "results"
    for k in (2, 4):
        (dataroot / f"{k}_players").mkdir(parents=True)
    row_data_dir = tmp_path / "curated"
    row_data_dir.mkdir()

    class _FailingFuture:
        def result(self):
            raise ValueError("boom")

    class _Executor:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def submit(self, *_args: object, **_kwargs: object):
            return _FailingFuture()

    monkeypatch.setattr(run_trueskill_module.cf, "ProcessPoolExecutor", _Executor)
    monkeypatch.setattr(run_trueskill_module.cf, "as_completed", lambda futures: list(futures))

    with pytest.raises(RuntimeError, match="TrueSkill block failed"):
        run_trueskill_module.run_trueskill(
            output_seed=11,
            root=tmp_path / "analysis",
            dataroot=dataroot,
            row_data_dir=row_data_dir,
            curated_rows_name="game_rows.parquet",
            workers=2,
            cell_freshness_sha256="a" * 64,
        )


def test_auxiliary_trueskill_exports_receive_sidecars(tmp_path: Path) -> None:
    context = _root_context(tmp_path, 11, ks=(2,))
    rating = _write_rating_cell(context, 2)
    suffix = "_seed11"
    paths = run_trueskill_module._rating_artifact_paths(
        context.config.trueskill_stage_dir, "2", suffix
    )
    paths["json"].write_text(
        '{"A":{"mu":30.112,"sigma":2.0},"B":{"mu":20.0,"sigma":3.0}}',
        encoding="utf-8",
    )
    shard, _done = run_trueskill_module._block_shard_paths(
        context.config.trueskill_stage_dir, "2", suffix
    )
    pq.write_table(pq.read_table(rating), shard)

    run_trueskill_module._ensure_auxiliary_rating_sidecars(
        context.config,
        cell=ScreeningRatingCell(root_seed=11, k=2, ratings_path=rating),
        suffix=suffix,
    )

    validate_artifact_sidecar(
        paths["json"], expected={"operation": "export_sequential_ratings_json"}
    )
    validate_artifact_sidecar(shard, expected={"operation": "snapshot_sequential_rating_cell"})
