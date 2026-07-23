"""
Fast, side-effect-free sanity checks for run_tournament.py
(all run in < 1 s).

We monkey-patch the heavy helpers so no real games are played.
"""

from __future__ import annotations

import json
import logging
import pickle
import types  # noqa: F401
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np  # noqa: F401 | Potentially imports something that needs it
import pytest

pytest.importorskip("pydantic")
pytest.importorskip("pyarrow")

import farkle.simulation.run_tournament as rt
from farkle.simulation.strategies import ThresholdStrategy
from farkle.simulation.workload_planner import plan_tournament_workload

# --------------------------------------------------------------------------- #
# Mini test doubles ? replace expensive pieces with cheap determinism
# --------------------------------------------------------------------------- #


def _mini_strats(n: int = 6):
    """Return deterministic Strategy objects with distinct __str__()."""
    return [ThresholdStrategy(50 + 50 * i, i % 3, True, True) for i in range(n)]


def fake_play_shuffle(task: rt.ShuffleTask | int) -> Counter[str]:
    # pretend player at index (seed % len(strats)) always wins
    seed = task.shuffle_seed if isinstance(task, rt.ShuffleTask) else task
    strats = _mini_strats(12)
    return Counter({str(strats[seed % len(strats)]): 1})


@pytest.fixture(autouse=True)
def fast_helpers(monkeypatch):
    """
    Keep tests lightning-fast **and** remember the real `_play_shuffle` so
    individual tests can restore it when they *need* the real behavior.
    """
    strats = _mini_strats(12)
    monkeypatch.setattr(
        rt, "generate_strategy_grid", lambda *a, **kw: (strats, None), raising=True  # noqa: ARG005
    )

    # save the genuine function on the module *once*
    if not hasattr(rt, "_ORIG_PLAY_SHUFFLE"):
        rt._ORIG_PLAY_SHUFFLE = rt._play_shuffle  # type: ignore

    monkeypatch.setattr(rt, "_play_shuffle", fake_play_shuffle, raising=True)


@pytest.mark.parametrize(
    ("n_players", "pattern"),
    [
        (1, r"n_players must be ≥2"),
        (8, r"n_players must divide [\d,]+"),
    ],
)
def test_run_tournament_invalid_player_counts(n_players: int, pattern: str) -> None:
    cfg = rt.TournamentConfig(n_players=n_players)
    cfg.num_shuffles = 1

    with pytest.raises(ValueError, match=pattern):
        rt.run_tournament(config=cfg)


def test_init_worker_rejects_bad_player_counts(monkeypatch) -> None:
    strats = _mini_strats(3)
    cfg = rt.TournamentConfig(n_players=7)
    monkeypatch.setattr(rt, "_STATE", None, raising=False)

    with pytest.raises(ValueError, match=r"n_players must divide [\d,]+"):
        rt._init_worker(strats, cfg)


def test_run_chunk_logs_and_propagates(monkeypatch, caplog) -> None:
    class BoomError(RuntimeError):
        pass

    def boom(_seed: int):
        raise BoomError("boom")

    monkeypatch.setattr(rt, "_play_shuffle", boom, raising=True)

    with caplog.at_level(logging.ERROR, logger=rt.LOGGER.name), pytest.raises(BoomError):
        rt._run_chunk([123])

    assert any("Shuffle failed" in rec.getMessage() for rec in caplog.records)
    logged = [rec for rec in caplog.records if "Shuffle failed" in rec.getMessage()][0]
    assert logged.stage == "simulation"
    assert logged.shuffle_seed == 123


def test_coerce_counter_and_metric_sums_helpers() -> None:
    wins = rt._coerce_counter({"1": "2", "A": 3})
    assert wins == Counter({1: 2, "A": 3})

    bad_metric_payload: Any = {"winning_score": {"5": "4.5"}}
    sums = rt._coerce_metric_sums(bad_metric_payload)
    assert sums is not None
    assert sums["winning_score"][5] == pytest.approx(4.5)
    assert sums["winner_rolls"] == {}


def test_coerce_counter_rejects_invalid_payload() -> None:
    with pytest.raises(TypeError, match="Unexpected win_totals payload type"):
        rt._coerce_counter([("A", 1)])


def test_measure_throughput_uses_spawned_seeds(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: list[int] = []

    monkeypatch.setattr(rt.urandom, "spawn_seeds", lambda n, seed=0: [11, 22, 33], raising=True)
    monkeypatch.setattr(
        rt,
        "_play_game",
        lambda seed, strategies, **_kwargs: observed.append(seed) or {"winner_seat": "p0"},
        raising=True,
    )

    perf_state = {"idx": 0, "vals": [10.0, 10.5]}

    def fake_perf() -> float:
        value = perf_state["vals"][perf_state["idx"]]
        perf_state["idx"] += 1
        return value

    monkeypatch.setattr(rt.time, "perf_counter", fake_perf, raising=True)

    value = rt._measure_throughput(_mini_strats(2), sample_games=3, seed=7)
    assert value == pytest.approx(6.0)
    assert observed == [11, 22, 33]


def test_manifest_int_set_parses_valid_values(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        "\n".join(
            [
                '{"shuffle_seed": "5"}',
                '{"shuffle_seed": "bad"}',
                '{"shuffle_seed": 3}',
                '{"other": 9}',
            ]
        )
        + "\n"
    )

    assert rt._manifest_int_set(manifest_path, "shuffle_seed") == {3, 5}


def test_run_tournament_non_metrics_chunk_execution_and_corrupt_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    strats = _mini_strats(4)
    monkeypatch.setattr(rt, "generate_strategy_grid", lambda *a, **k: (strats, None), raising=True)
    monkeypatch.setattr(rt, "_measure_throughput", lambda sample: 8.0, raising=True)
    monkeypatch.setattr(
        rt.urandom,
        "coordinate_seed",
        lambda _purpose, *, root_seed, shuffle_index=0, **_kwargs: root_seed + shuffle_index,
        raising=True,
    )

    chunk_calls: list[list[int]] = []

    def fake_process_map(func, iterable, *, initializer=None, initargs=(), **kwargs):
        if initializer is not None:
            initializer(*initargs)
        for item in iterable:
            chunk_calls.append([task.shuffle_seed for task in item[1]])
            yield func(item)

    monkeypatch.setattr(rt.parallel, "process_map", fake_process_map)
    monkeypatch.setattr(
        rt,
        "_play_shuffle",
        lambda task: Counter({f"W{task.shuffle_seed}": 1}),
        raising=True,
    )

    cfg = rt.TournamentConfig(
        n_players=2,
        num_shuffles=5,
        desired_sec_per_chunk=1,
        ckpt_every_sec=999,
        deterministic_batch_size=1,
    )
    ckpt = tmp_path / "checkpoint.pkl"
    plan_path = tmp_path / "workload.json"
    workload_plan = plan_tournament_workload(
        root_seed=0,
        k=2,
        strategy_count=4,
        resolution_delta=0.9,
        batch_count=5,
        min_shuffles_per_batch=1,
    )
    rt.run_tournament(
        config=cfg,
        num_shuffles=cfg.num_shuffles,
        checkpoint_path=ckpt,
        n_jobs=1,
        collect_metrics=False,
        workload_plan=workload_plan,
        workload_plan_path=plan_path,
    )

    assert chunk_calls == [[0], [1], [2], [3], [4]]
    payload = pickle.loads(ckpt.read_bytes())
    assert payload["win_totals"] == Counter({"W0": 1, "W1": 1, "W2": 1, "W3": 1, "W4": 1})
    plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan_payload["projected_games_per_second"] == 8.0
    assert plan_payload["projected_runtime_seconds"] == pytest.approx(10 / 8)

    ckpt.write_bytes(pickle.dumps(123))
    with pytest.raises(AttributeError):
        rt.run_tournament(
            config=cfg,
            num_shuffles=cfg.num_shuffles,
            checkpoint_path=ckpt,
            n_jobs=1,
            collect_metrics=False,
        )


def test_checkpoint_coordinates_resume_without_row_or_metric_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    strats = _mini_strats(4)
    monkeypatch.setattr(rt, "generate_strategy_grid", lambda *a, **k: (strats, None))
    monkeypatch.setattr(rt, "_measure_throughput", lambda sample: 8.0)
    monkeypatch.setattr(
        rt.urandom,
        "coordinate_seed",
        lambda _purpose, *, root_seed, shuffle_index=0, **_kwargs: root_seed + shuffle_index,
    )

    def completed_shuffle(task: rt.ShuffleTask) -> rt.OutcomeCounter:
        winner = f"W{task.shuffle_index}"
        loser = f"L{task.shuffle_index}"
        counts = rt.OutcomeCounter({winner: 1})
        counts.attempted_exposures.update({winner: 1, loser: 1})
        counts.completed_exposures.update({winner: 1, loser: 1})
        counts.games_attempted = 1
        counts.games_completed = 1
        return counts

    monkeypatch.setattr(rt, "_play_shuffle", completed_shuffle)

    cfg = rt.TournamentConfig(
        n_players=2,
        num_shuffles=3,
        ckpt_every_sec=0,
        deterministic_batch_size=1,
    )
    ckpt = tmp_path / "checkpoint.pkl"

    def interrupt_after_first(func, iterable, *, initializer=None, initargs=(), **kwargs):
        if initializer is not None:
            initializer(*initargs)
        for position, item in enumerate(iterable):
            if position == 1:
                raise KeyboardInterrupt
            yield func(item)

    monkeypatch.setattr(rt.parallel, "process_map", interrupt_after_first)
    with pytest.raises(KeyboardInterrupt):
        rt.run_tournament(
            config=cfg,
            num_shuffles=cfg.num_shuffles,
            checkpoint_path=ckpt,
            n_jobs=1,
        )

    interrupted = pickle.loads(ckpt.read_bytes())
    assert interrupted["win_totals"] == Counter({"W0": 1})
    assert interrupted["meta"]["completed_shuffle_indices"] == [0]
    assert interrupted["meta"]["completed_process_block_indices"] == [1]

    resumed_blocks: list[list[int]] = []

    def resume_map(func, iterable, *, initializer=None, initargs=(), **kwargs):
        if initializer is not None:
            initializer(*initargs)
        for item in iterable:
            resumed_blocks.append([task.shuffle_index for task in item[1]])
            yield func(item)

    monkeypatch.setattr(rt.parallel, "process_map", resume_map)
    rt.run_tournament(
        config=cfg,
        num_shuffles=cfg.num_shuffles,
        checkpoint_path=ckpt,
        n_jobs=1,
    )

    completed = pickle.loads(ckpt.read_bytes())
    assert resumed_blocks == [[1], [2]]
    assert completed["win_totals"] == Counter({"W0": 1, "W1": 1, "W2": 1})
    assert completed["meta"]["completed_shuffle_indices"] == [0, 1, 2]
    assert completed["meta"]["completed_process_block_indices"] == [1, 2, 3]

    uninterrupted_path = tmp_path / "uninterrupted.pkl"
    rt.run_tournament(
        config=cfg,
        num_shuffles=cfg.num_shuffles,
        checkpoint_path=uninterrupted_path,
        n_jobs=1,
    )
    uninterrupted = pickle.loads(uninterrupted_path.read_bytes())
    assert completed["win_totals"] == uninterrupted["win_totals"]
    assert completed["outcome_counts"] == uninterrupted["outcome_counts"]
    assert (
        completed["meta"]["completed_shuffle_indices"]
        == uninterrupted["meta"]["completed_shuffle_indices"]
    )


def test_direct_resume_rejects_v1_checkpoint_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    strategies = _mini_strats(4)
    monkeypatch.setattr(rt, "_measure_throughput", lambda _sample: 1.0)
    checkpoint = tmp_path / "v1.pkl"
    checkpoint.write_bytes(
        pickle.dumps({"win_totals": Counter(), "meta": {"rng_scheme_version": 1}})
    )

    with pytest.raises(ValueError, match="Checkpoint RNG scheme is stale"):
        rt.run_tournament(
            config=rt.TournamentConfig(n_players=2, num_shuffles=1),
            checkpoint_path=checkpoint,
            strategies=strategies,
            num_shuffles=1,
            n_jobs=1,
        )


def test_resume_ownership_uses_shuffle_index_not_scalar_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rt.urandom, "coordinate_seed", lambda *args, **kwargs: 77)

    pending = list(
        rt._iter_pending_chunk_items(
            num_shuffles=2,
            shuffles_per_chunk=2,
            global_seed=32,
            k=2,
            deterministic_batch_size=2,
            completed_shuffle_indices={0},
            completed_chunk_indices=set(),
        )
    )

    assert [(task.shuffle_index, task.shuffle_seed) for task in pending[0][1]] == [(1, 77)]


def test_shuffle_rows_preserve_turns_and_rng_coordinates() -> None:
    strategies = _mini_strats(4)
    config = rt.TournamentConfig(n_players=2, n_strategies=4)
    rt._init_worker(strategies, config)
    task = rt.ShuffleTask(
        root_seed=91,
        k=2,
        shuffle_index=7,
        shuffle_seed=1234,
        deterministic_batch_id=2,
    )

    _, _, _, rows = rt._play_one_shuffle(task, collect_rows=True)

    assert len(rows) == 2
    assert [row["game_index"] for row in rows] == [0, 1]
    for row in rows:
        assert row["root_seed"] == 91
        assert row["k"] == 2
        assert row["shuffle_index"] == 7
        assert row["deterministic_batch_id"] == 2
        assert row["shuffle_seed"] == 1234
        assert row["rng_scheme_version"] == rt.urandom.RNG_SCHEME_VERSION
        assert row["rng_purpose_namespace"] == 102
        assert row["P1_n_turns"] >= 1
        assert row["P2_n_turns"] >= 1
