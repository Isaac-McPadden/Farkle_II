"""Authenticated bounded seat behavior and game-level turn opportunity products."""

from __future__ import annotations

import heapq
import json
import math
import shutil
from collections.abc import Callable, Generator, Iterator
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from farkle.analysis.outcome_precision import _read_span, _validate_support
from farkle.analysis.outcome_publication import (
    _admit_source_footers,
    _checked,
    _CoalescedWriter,
    _dependent_unit,
    _Design,
    _identity_from_json,
    _resolve_design,
    _source_units,
    _unit_from_json,
    _write_rows,
    outcome_is_up_to_date,
)
from farkle.config import AppConfig
from farkle.utils.arrow_batches import iter_parquet_tables_by_bytes
from farkle.utils.artifact_contract import (
    ArtifactSidecar,
    make_artifact_sidecar,
    sha256_file,
    sidecar_path,
    validate_artifact_sidecar,
)
from farkle.utils.artifacts import write_json_artifact_atomic, write_parquet_artifact_atomic
from farkle.utils.authenticated_contract import identity_sha256
from farkle.utils.partitioned_stage import (
    PartitionedStageIdentity,
    PartitionedUnit,
    resolved_code_identity_sha256,
    run_partitioned_stage,
    validate_final_manifest,
)
from farkle.utils.release_identity import CapturedV3Inputs, capture_v3_inputs
from farkle.utils.schema_helpers import expected_schema_for
from farkle.utils.stage_completion import stage_done_path, stage_is_up_to_date, write_stage_done
from farkle.utils.writer import ParquetShardWriter

METHOD_VERSION = 1
BEHAVIORS = (
    "rolls",
    "farkles",
    "hot_dice",
    "highest_turn",
    "smart_one_uses",
    "n_smart_one_dice",
    "smart_five_uses",
    "n_smart_five_dice",
)
COUNTS = (
    "exposures",
    "zero_turn_exposures",
    "zero_round_exposures",
    "turn_round_mismatch_count",
    "score_sum",
    "turns_sum",
    "rounds_sum",
    *(f"{n}_sum" for n in BEHAVIORS),
)
RATIOS = ("score_per_turn_sum", "score_per_round_sum")
ROOT: list[pa.Field] = [pa.field("root_seed", pa.int64(), False), pa.field("k", pa.int16(), False)]
BATCH = pa.field("deterministic_batch_id", pa.int32(), False)
SEAT: list[pa.Field] = [
    pa.field("strategy", pa.int32(), False),
    pa.field("seat", pa.int16(), False),
]
VALUES: list[pa.Field] = [
    *(pa.field(n, pa.int64(), False) for n in COUNTS[:7]),
    *(pa.field(n, pa.float64(), False) for n in RATIOS),
    *(pa.field(n, pa.int64(), False) for n in COUNTS[7:]),
]
B7 = pa.schema([*ROOT, BATCH, *SEAT, pa.field("termination_status", pa.string(), False), *VALUES])
Q7 = pa.schema(
    [
        *ROOT,
        *SEAT,
        *(pa.field(n, pa.string(), False) for n in ("conditioning", "metric")),
        pa.field("numerator", pa.float64(), False),
        pa.field("denominator", pa.int64(), False),
        pa.field("estimate", pa.float64()),
        pa.field("status", pa.string(), False),
    ]
)
O7_NAMES = (
    "attempted_games",
    "completed_games",
    "safety_games",
    "completed_turn_range_sum",
    "completed_total_turns",
)
O7 = pa.schema([*ROOT, BATCH, *(pa.field(n, pa.int64(), False) for n in O7_NAMES)])
OPPORTUNITY_SCHEMA = pa.schema(
    [
        *ROOT,
        pa.field("seat", pa.int16(), False),
        pa.field("completed_games", pa.int64(), False),
        pa.field("turns_sum", pa.int64()),
        *(
            pa.field(n, pa.float64())
            for n in ("mean_turns", "turn_share", "mean_turn_range", "turn_share_imbalance")
        ),
        pa.field("status", pa.string(), False),
    ]
)
OA = pa.schema(
    [
        pa.field("root_seed", pa.int64(), False),
        pa.field("required_ks", pa.list_(pa.int16()), False),
        pa.field("weights", pa.list_(pa.float64()), False),
        pa.field("mean_turn_range_equal_k", pa.float64()),
        pa.field("status", pa.string(), False),
    ]
)
# Uncombined float terms preserve canonical row order across source/merge layouts.
RUN = pa.schema(
    [
        BATCH,
        *SEAT,
        pa.field("termination_status", pa.string(), False),
        pa.field("ordinal", pa.int64(), False),
        *VALUES,
    ]
)
RUN_KEYS = ("deterministic_batch_id", "strategy", "seat", "termination_status", "ordinal")

RUN_DTYPE = np.dtype(
    [
        (n, "U12" if n == "termination_status" else "<f8" if n in RATIOS else "<i8")
        for n in RUN.names
    ]
)


def behavior_paths(cfg: AppConfig, k: int) -> tuple[Path, ...]:
    root = cfg.seat_behavior_units_dir(k)
    return (
        cfg.seat_behavior_batch_path(k),
        cfg.seat_behavior_summaries_path(k),
        cfg.turn_opportunity_batch_path(k),
        cfg.turn_opportunity_path(k),
        root / "plan.json",
        root / "publication.json",
    )


def _inputs(cfg: AppConfig, k: int) -> list[Path]:
    return [
        cfg.ingested_rows_curated(k),
        cfg.outcome_batch_stats_path(k),
        cfg.outcome_batch_inventory_path(k),
        cfg.outcome_support_path(k),
        cfg.outcome_checkpoint_inventory_path(k),
        cfg.metrics_all_player_batch_path(k),
        cfg.strategy_manifest_root_path(),
        cfg.simulation_workload_plan_path(k),
    ]


def _metadata(
    cfg: AppConfig,
    k: int,
    path: Path,
    operation: str,
    sources: list[Path],
    schema: pa.Schema | None = None,
) -> ArtifactSidecar:
    return make_artifact_sidecar(
        cfg,
        path,
        producer="metrics",
        scope="by_k",
        source_scope="by_k",
        operation=operation,
        weighted_quantity="terminal_behavior_and_actual_turn_opportunity",
        support_count_role="certified_player_exposures_and_once_per_game_opportunities",
        uncertainty_method="none",
        replication_unit="deterministic_shuffle_batch",
        conditioning="explicit_completed_safety_limit_or_all_attempted",
        source_artifacts=sources,
        player_counts=[k],
        required_player_counts=[k],
        missing_cell_policy="fail",
        consistency_columns=[] if schema is None else schema.names,
        method_contract={
            "kind": "operation",
            "procedure": operation,
            "parameters": {
                "seat_behavior_method_version": 1,
                "turn_opportunity_method_version": 1,
                "game_ratio_order": "canonical_row_then_seat",
                "proxy_gap": "exact_minus_round_proxy",
                "zero_safety_denominator": "zero_score_ratio_zero",
            },
        },
    )


def _rows(path: Path, schema: pa.Schema, budget: int) -> Generator[dict[str, Any], None, None]:
    if not pq.read_schema(path).remove_metadata().equals(schema):
        raise ValueError(f"incompatible behavior schema: {path}")
    for _, _, table in iter_parquet_tables_by_bytes(
        path,
        columns=schema.names,
        max_batch_bytes=max(1, budget // 4),
        max_batch_rows=max(1, budget // 16384),
        use_threads=False,
    ):
        yield from table.to_pylist()


def _zero() -> dict[str, Any]:
    return {**dict.fromkeys(COUNTS, 0), **dict.fromkeys(RATIOS, 0.0)}


def _add(total: dict[str, Any], row: dict[str, Any]) -> None:
    for n in COUNTS:
        total[n] = _checked(total[n] + row[n])
    for n in RATIOS:
        total[n] += row[n]
        if not math.isfinite(total[n]):
            raise OverflowError("nonfinite behavior ratio sum")


def behavior_records(
    row: dict[str, Any], *, root: int, k: int, batch: int, ordinal: int
) -> Iterator[dict[str, Any]]:
    """One record per actual exposure and one game record; never deduplicate seats."""
    if (row["root_seed"], row["k"], row["deterministic_batch_id"]) != (root, k, batch):
        raise ValueError("behavior source root/k/batch mismatch")
    status = row["termination_status"]
    if status not in ("completed", "safety_limit"):
        raise ValueError("unknown behavior termination")
    rounds = row["n_rounds"]
    if type(rounds) is not int or rounds < 0 or (status == "completed" and not rounds):
        raise ValueError("invalid actual round count")
    turns = []
    for seat in range(1, k + 1):

        def value(n: str, seat: int = seat) -> int:
            v = row[f"P{seat}_{n}"]
            if type(v) is not int:
                raise ValueError(f"missing/noninteger actual-seat {n}")
            return _checked(v)

        score, turn, sid = value("score"), value("n_turns"), value("strategy")
        if (status == "completed" and turn == 0) or ((turn == 0 or rounds == 0) and score != 0):
            raise ValueError("invalid score/turn/round denominator")
        turns.append(turn)
        yield {
            "deterministic_batch_id": batch,
            "strategy": sid,
            "seat": seat,
            "termination_status": status,
            "ordinal": ordinal,
            "exposures": 1,
            "zero_turn_exposures": int(turn == 0),
            "zero_round_exposures": int(rounds == 0),
            "turn_round_mismatch_count": int(turn != rounds),
            "score_sum": score,
            "turns_sum": turn,
            "rounds_sum": rounds,
            "score_per_turn_sum": score / turn if turn else 0.0,
            "score_per_round_sum": score / rounds if rounds else 0.0,
            **{f"{n}_sum": value(n) for n in BEHAVIORS},
        }
    # Internal sentinel: score_sum stores range; turns_sum stores all-seat turns.
    yield {
        **_zero(),
        "deterministic_batch_id": batch,
        "strategy": -1,
        "seat": 0,
        "termination_status": status,
        "ordinal": ordinal,
        "exposures": 1,
        "score_sum": max(turns) - min(turns) if status == "completed" else 0,
        "turns_sum": _checked(sum(turns)) if status == "completed" else 0,
    }


@dataclass(frozen=True)
class _SourceWriter:
    design: _Design
    budget: int

    def __call__(self, unit: PartitionedUnit, path: Path) -> None:
        batch, _, start, stop, rg, rg_start = map(int, unit.key)
        d = self.design
        columns = [
            "root_seed",
            "k",
            "deterministic_batch_id",
            "shuffle_index",
            "game_index",
            "termination_status",
            "n_rounds",
            *(
                f"P{s}_{n}"
                for s in range(1, d.k + 1)
                for n in ("strategy", "score", "n_turns", *BEHAVIORS)
            ),
        ]
        schema = pq.read_schema(d.source)
        expected = expected_schema_for(d.k)
        if any(schema.field(n).type != expected.field(n).type for n in columns):
            raise ValueError("incompatible behavior source types")
        records = np.empty((stop - start) * (d.k + 1), dtype=RUN_DTYPE)
        used = 0
        position = rg_start
        for _, _, table in iter_parquet_tables_by_bytes(
            Path(d.source),
            columns=columns,
            max_batch_bytes=self.budget // 4,
            max_batch_rows=max(1, self.budget // (32768 * (d.k + 1))),
            start_row_group=rg,
            use_threads=False,
        ):
            end = position + table.num_rows
            left, right = max(start, position), min(stop, end)
            if right > left:
                for ordinal, row in enumerate(
                    table.slice(left - position, right - left).to_pylist(), left
                ):
                    if (row["shuffle_index"], row["game_index"]) != divmod(
                        ordinal, d.games_per_shuffle
                    ):
                        raise ValueError("missing, duplicate or reordered behavior coordinate")
                    for record in behavior_records(
                        row, root=d.root, k=d.k, batch=batch, ordinal=ordinal
                    ):
                        records[used] = tuple(record[n] for n in RUN.names)
                        used += 1
            position = end
            if position >= stop:
                break
        if position < stop:
            raise ValueError("behavior source ended before slice boundary")
        records = records[:used]
        records.sort(order=RUN_KEYS, kind="stable")
        _write_rows(path, RUN, ({n: r[n].item() for n in RUN.names} for r in records), self.budget)


@dataclass(frozen=True)
class _MergeWriter:
    budget: int

    def __call__(self, unit: PartitionedUnit, path: Path) -> None:
        paths = [Path(str(p)) for p in unit.key[2:]]
        if not 1 <= len(paths) <= 16:
            raise ValueError("invalid behavior merge fan-in")
        streams = [_rows(p, RUN, self.budget // len(paths)) for p in paths]
        try:
            _write_rows(
                path,
                RUN,
                heapq.merge(*streams, key=lambda r: tuple(r[n] for n in RUN_KEYS)),
                self.budget,
            )
        finally:
            for stream in streams:
                stream.close()


@dataclass(frozen=True)
class _BatchWriter:
    cfg: AppConfig
    design: _Design
    budget: int

    def __call__(self, unit: PartitionedUnit, path: Path) -> None:
        batch = int(unit.key[0])
        d = self.design
        rows = iter(_rows(Path(str(unit.key[2])), RUN, self.budget))
        current = next(rows, None)
        opportunity: dict[str, int] = dict.fromkeys(O7_NAMES, 0)
        while current is not None and current["strategy"] == -1:
            completed = current["termination_status"] == "completed"
            for name, count in (
                ("attempted_games", 1),
                ("completed_games", int(completed)),
                ("safety_games", int(not completed)),
                ("completed_turn_range_sum", current["score_sum"]),
                ("completed_total_turns", current["turns_sum"]),
            ):
                opportunity[name] = _checked(opportunity[name] + count)
            current = next(rows, None)
        if opportunity["attempted_games"] != d.games_per_batch:
            raise ValueError("behavior game support mismatch")
        expected_rows = iter(
            row
            for t in _read_span(
                pq.ParquetFile(self.cfg.outcome_batch_stats_path(d.k)),
                batch * len(d.roster) * d.k,
                (batch + 1) * len(d.roster) * d.k,
                self.budget // 4,
            )
            for row in t.to_pylist()
        )

        def certified():
            nonlocal current
            for sid, _ in d.roster:
                for seat in range(1, d.k + 1):
                    expected = next(expected_rows, None)
                    if expected is None or (
                        expected["deterministic_batch_id"],
                        expected["strategy"],
                        expected["seat"],
                    ) != (batch, sid, seat):
                        raise ValueError("missing certified outcome support cell")
                    for status in ("completed", "safety_limit"):
                        total = _zero()
                        previous = -1
                        key = (batch, sid, seat, status)
                        while (
                            current is not None and tuple(current[n] for n in RUN_KEYS[:-1]) == key
                        ):
                            if current["ordinal"] <= previous:
                                raise ValueError("duplicate or reordered exposure")
                            previous = current["ordinal"]
                            _add(total, current)
                            current = next(rows, None)
                        if (
                            total["exposures"]
                            != expected["completed" if status == "completed" else "safety"]
                        ):
                            raise ValueError(
                                "behavior termination support differs from certified outcomes"
                            )
                        yield {
                            "root_seed": d.root,
                            "k": d.k,
                            "deterministic_batch_id": batch,
                            "strategy": sid,
                            "seat": seat,
                            "termination_status": status,
                            **total,
                        }
            if current is not None or next(expected_rows, None) is not None:
                raise ValueError("unexpected behavior support cell")

        _write_rows(
            path,
            B7,
            certified(),
            self.budget,
            {
                "turn_opportunity": json.dumps(
                    {"root_seed": d.root, "k": d.k, "deterministic_batch_id": batch, **opportunity},
                    sort_keys=True,
                )
            },
        )


def summary_rows(
    root: int, k: int, sid: int, seat: int, conditioning: str, v: dict[str, Any]
) -> Iterator[dict[str, Any]]:
    metrics = {
        "points_per_turn": (v["score_sum"], v["turns_sum"]),
        "mean_game_points_per_turn": (v["score_per_turn_sum"], v["exposures"]),
        "mean_game_points_per_round": (v["score_per_round_sum"], v["exposures"]),
        "round_proxy_gap": (v["score_per_turn_sum"] - v["score_per_round_sum"], v["exposures"]),
    }
    for metric, field in (
        ("turn_round_mismatch_rate", "turn_round_mismatch_count"),
        ("zero_turn_rate", "zero_turn_exposures"),
        ("zero_round_rate", "zero_round_exposures"),
        ("mean_turns", "turns_sum"),
        ("mean_rounds", "rounds_sum"),
        ("mean_score", "score_sum"),
        ("mean_highest_turn", "highest_turn_sum"),
    ):
        metrics[metric] = (v[field], v["exposures"])
    for n in BEHAVIORS:
        if n != "highest_turn":
            metrics[f"{n}_per_turn"] = (v[f"{n}_sum"], v["turns_sum"])
            metrics[f"{n}_per_exposure"] = (v[f"{n}_sum"], v["exposures"])
    for n in ("one", "five"):
        metrics[f"smart_{n}_dice_per_use"] = (v[f"n_smart_{n}_dice_sum"], v[f"smart_{n}_uses_sum"])
    for metric, (numerator, denominator) in sorted(metrics.items()):
        yield {
            "root_seed": root,
            "k": k,
            "strategy": sid,
            "seat": seat,
            "conditioning": conditioning,
            "metric": metric,
            "numerator": float(numerator),
            "denominator": denominator,
            "estimate": numerator / denominator if denominator else None,
            "status": "ok" if denominator else "no_denominator",
        }


@dataclass(frozen=True)
class _SummaryWriter:
    cfg: AppConfig
    design: _Design
    tile_size: int
    budget: int

    def __call__(self, unit: PartitionedUnit, path: Path) -> None:
        tile = int(unit.key[0])
        d = self.design
        roster = d.roster[tile * self.tile_size : (tile + 1) * self.tile_size]
        sums = {
            (sid, seat, status): _zero()
            for sid, _ in roster
            for seat in range(d.k + 1)
            for status in ("completed", "safety_limit")
        }
        ids = {sid for sid, _ in roster}
        count = 0
        for row in _rows(self.cfg.seat_behavior_batch_path(d.k), B7, self.budget):
            sid, seat, status = row["strategy"], row["seat"], row["termination_status"]
            if sid in ids:
                _add(sums[sid, seat, status], row)
                _add(sums[sid, 0, status], row)
                count += 1
        if count != len(ids) * d.k * 2 * d.batches:
            raise ValueError("behavior summary lacks full batch support")

        def records():
            for sid, _ in roster:
                for seat in range(d.k + 1):
                    all_attempts = _zero()
                    for status in ("completed", "safety_limit"):
                        value = sums[sid, seat, status]
                        _add(all_attempts, value)
                        yield from summary_rows(d.root, d.k, sid, seat, status, value)
                    yield from summary_rows(d.root, d.k, sid, seat, "all_attempted", all_attempts)

        _write_rows(path, Q7, records(), self.budget)


@dataclass(frozen=True)
class _Sidecars:
    cfg: AppConfig
    k: int
    sources: tuple[str, ...]
    schema: pa.Schema
    captured: CapturedV3Inputs | None = None
    dependent: bool = False

    def __call__(self, unit: PartitionedUnit, path: Path) -> ArtifactSidecar:
        sources = [Path(p) for p in self.sources]
        if self.dependent:
            sources += [Path(str(p)) for p in unit.key[2:]]
        metadata = _metadata(self.cfg, self.k, path, "seat_behavior_unit", sources, self.schema)
        return replace(metadata, _captured_v3_inputs=self.captured) if self.captured else metadata


def _validate_unit(_unit: PartitionedUnit, path: Path) -> dict[str, str]:
    metadata = validate_artifact_sidecar(path, expected={"operation": "seat_behavior_unit"})
    if metadata.consistency_columns != pq.read_schema(path).names:
        raise ValueError(f"incomplete behavior checkpoint column inventory: {path}")
    return {"sidecar_sha256": sha256_file(sidecar_path(path))}


def validate_behavior_checkpoints(publication: Path) -> bool:
    try:
        validate_artifact_sidecar(publication)
        payload = json.loads(publication.read_text(encoding="utf-8"))
        validate_artifact_sidecar(publication.parent / "plan.json")
        if payload["method_version"] != METHOD_VERSION or not payload["stages"]:
            return False
        names = set()
        for descriptor in payload["stages"]:
            name = descriptor["name"]
            if (
                name in names
                or not name
                or any(c not in "abcdefghijklmnopqrstuvwxyz_0123456789" for c in name)
            ):
                return False
            names.add(name)
            root = publication.parent / name
            units = [_unit_from_json(v) for v in descriptor["units"]]
            identity = _identity_from_json(descriptor["identity"])
            if dict(identity.input_identities).get("plan") != sha256_file(
                publication.parent / "plan.json"
            ):
                return False
            if (
                validate_final_manifest(
                    root / "manifest.jsonl",
                    root=root,
                    identity=identity,
                    unit_source=units.__iter__,
                    require_sidecar=True,
                    validator=_validate_unit,
                )
                is None
            ):
                return False
        return {"source", "certified_batches", "summaries"} <= names
    except (OSError, RuntimeError, ValueError, KeyError, TypeError):
        return False


def behavior_is_up_to_date(cfg: AppConfig, k: int) -> bool:
    paths = behavior_paths(cfg, k)
    return stage_is_up_to_date(
        stage_done_path(cfg.metrics_per_k_dir(k), "seat_behavior"),
        inputs=_inputs(cfg, k),
        outputs=paths,
        cfg=cfg,
        stage="metrics",
        sidecar_artifacts=paths,
    ) and validate_behavior_checkpoints(paths[-1])


def _publish_rows(
    cfg: AppConfig,
    k: int,
    path: Path,
    schema: pa.Schema,
    records: Iterator[dict[str, Any]],
    sources: list[Path],
) -> None:
    budget = cfg.resources.stage_batch_bytes["analysis"]
    with ParquetShardWriter(
        str(path),
        schema=schema,
        sidecar=_metadata(cfg, k, path, "seat_behavior_v1", sources, schema),
        temp_dir=cfg.io.temp_dir,
    ) as writer:
        coalescer = _CoalescedWriter(writer.write_batch, schema, budget // 4)
        buffer = []
        for row in records:
            buffer.append(row)
            if len(buffer) >= max(1, budget // 16384):
                coalescer.write_batch(pa.Table.from_pylist(buffer, schema=schema))
                buffer.clear()
        if buffer:
            coalescer.write_batch(pa.Table.from_pylist(buffer, schema=schema))
        coalescer.flush()


def build_seat_behavior(cfg: AppConfig, k: int, *, force: bool = False) -> tuple[Path, ...]:
    if k not in cfg.sim.n_players_list:
        raise ValueError("behavior k outside configured support")
    if not force and behavior_is_up_to_date(cfg, k) and outcome_is_up_to_date(cfg, k):
        return behavior_paths(cfg, k)
    _admit_source_footers(cfg)
    if not outcome_is_up_to_date(cfg, k):
        raise ValueError("behavior requires authenticated complete outcome support")
    d = _resolve_design(cfg, k)
    budget = cfg.resources.stage_batch_bytes["analysis"]
    _validate_support(cfg, d, budget)
    inputs = _inputs(cfg, k)
    for p in inputs:
        validate_artifact_sidecar(p)
    root = cfg.seat_behavior_units_dir(k)
    plan_path = root / "plan.json"
    design_hash = identity_sha256(
        {
            "design": asdict(d),
            "inputs": [(str(p), sha256_file(p), sha256_file(sidecar_path(p))) for p in inputs],
        }
    )
    code_hash = resolved_code_identity_sha256(cfg)
    rows_per_unit = budget // (4 * RUN_DTYPE.itemsize * (k + 1))
    tile_size = min(len(d.roster), budget // (4 * (k + 1) * 2 * 2048))
    if min(rows_per_unit, tile_size) < 1:
        raise MemoryError(
            "seat behavior needs a larger analysis stage_batch_bytes for one game/tile"
        )
    plan = None
    if plan_path.exists():
        try:
            validate_artifact_sidecar(plan_path)
            old = json.loads(plan_path.read_text(encoding="utf-8"))
            if (
                old["design_hash"] == design_hash
                and old["code_hash"] == code_hash
                and not (force and budget < old["minimum_budget"])
            ):
                plan = old
        except (OSError, RuntimeError, ValueError, KeyError):
            pass
    if plan is None:
        resolved = root.resolve()
        if resolved != cfg.metrics_per_k_dir(
            k
        ).resolve() / "_seat_behavior_units" or not resolved.is_relative_to(
            cfg.results_root.resolve()
        ):
            raise ValueError("behavior checkpoint path escaped owner")
        if resolved.exists():
            shutil.rmtree(resolved)
        plan = {
            "method_version": METHOD_VERSION,
            "design_hash": design_hash,
            "code_hash": code_hash,
            "minimum_budget": budget,
            "rows_per_unit": rows_per_unit,
            "tile_size": tile_size,
            "float_order": "canonical_game_order_within_batch_stratum",
        }
        write_json_artifact_atomic(
            plan,
            plan_path,
            sidecar=replace(
                _metadata(cfg, k, plan_path, "seat_behavior_plan", inputs),
                consistency_columns=sorted(plan),
            ),
        )
    if budget < plan["minimum_budget"]:
        raise MemoryError("behavior resume needs original allowance or --force to repartition")
    identity = PartitionedStageIdentity(
        "analysis",
        d.root,
        (("plan", sha256_file(plan_path)),),
        design_hash,
        code_hash,
        1,
        METHOD_VERSION,
    )
    descriptors: list[dict[str, Any]] = []
    manifests = []

    def run(
        name: str,
        units: list[PartitionedUnit],
        writer: Callable[[PartitionedUnit, Path], None],
        sources: list[Path],
        *,
        dependent: bool = False,
        schema: pa.Schema = RUN,
    ) -> list[Path]:
        descriptor = {
            "name": name,
            "identity": asdict(identity),
            "units": [asdict(u) for u in units],
        }
        if len(json.dumps([*descriptors, descriptor])) * 4 > budget:
            raise MemoryError("behavior checkpoint metadata exceeds byte allowance")
        directory = root / name
        sources = [plan_path, *sources]
        metadata = _metadata(
            cfg, k, directory / "manifest.jsonl", "seat_behavior_manifest", sources
        )
        factory = _Sidecars(
            cfg,
            k,
            tuple(map(str, sources)),
            schema,
            None if dependent else capture_v3_inputs(metadata),
            dependent,
        )
        run_partitioned_stage(
            root=directory,
            identity=identity,
            unit_source=units.__iter__,
            writer=writer,
            resources=cfg.resources,
            requested_workers=cfg.analysis.n_jobs,
            mp_start_method=cfg.analysis.mp_start_method,
            force=force,
            sidecar_factory=factory,
            validator=_validate_unit,
            manifest_path=directory / "manifest.jsonl",
            manifest_sidecar=metadata,
            temp_dir=cfg.io.temp_dir,
        )
        descriptors.append(descriptor)
        manifests.append(directory / "manifest.jsonl")
        return [directory / "units" / u.relative_output for u in units]

    unit_count = d.batches * (
        (d.games_per_batch + plan["rows_per_unit"] - 1) // plan["rows_per_unit"]
    )
    if unit_count * 4096 > budget:
        raise MemoryError("behavior slice inventory exceeds metadata allowance")
    units = [
        replace(u, relative_output=f"batch-{int(u.key[0]):08d}/slice-{int(u.key[1]):08d}.parquet")
        for u in _source_units(d, plan["rows_per_unit"])
    ]
    slices = run("source", units, _SourceWriter(d, budget), inputs)
    groups: dict[int, list[Path]] = {}
    for u, p in zip(units, slices, strict=True):
        groups.setdefault(int(u.key[0]), []).append(p)
    level = 0
    while any(len(ps) > 1 for ps in groups.values()):
        merged: dict[int, list[Path]] = {}
        merge_units: list[PartitionedUnit] = []
        for batch, ps in sorted(groups.items()):
            if len(ps) == 1:
                merged[batch] = ps
            else:
                merge_units.extend(
                    _dependent_unit(batch, i, ps[start : start + 16])
                    for i, start in enumerate(range(0, len(ps), 16))
                )
        paths = run(f"merge_{level:04d}", merge_units, _MergeWriter(budget), [], dependent=True)
        for u, p in zip(merge_units, paths, strict=True):
            merged.setdefault(int(u.key[0]), []).append(p)
        groups = merged
        level += 1
    certificates = run(
        "certified_batches",
        [_dependent_unit(b, 0, ps) for b, ps in sorted(groups.items())],
        _BatchWriter(cfg, d, budget),
        [cfg.outcome_batch_stats_path(k)],
        dependent=True,
        schema=B7,
    )
    _publish_rows(
        cfg,
        k,
        cfg.seat_behavior_batch_path(k),
        B7,
        (r for p in certificates for r in _rows(p, B7, budget)),
        certificates,
    )
    opportunities = [
        json.loads((pq.ParquetFile(p).metadata.metadata or {})[b"turn_opportunity"])
        for p in certificates
    ]
    _publish_rows(cfg, k, cfg.turn_opportunity_batch_path(k), O7, iter(opportunities), certificates)
    _reconcile_all_player(cfg, d, budget)
    summaries = run(
        "summaries",
        [
            PartitionedUnit((t,), f"tile-{t:08d}.parquet")
            for t in range((len(d.roster) + plan["tile_size"] - 1) // plan["tile_size"])
        ],
        _SummaryWriter(cfg, d, plan["tile_size"], budget),
        [cfg.seat_behavior_batch_path(k)],
        schema=Q7,
    )
    publication = root / "publication.json"
    write_json_artifact_atomic(
        {"method_version": METHOD_VERSION, "stages": descriptors},
        publication,
        sidecar=replace(
            _metadata(cfg, k, publication, "seat_behavior_inventory", manifests),
            consistency_columns=["method_version", "stages"],
        ),
    )
    if not validate_behavior_checkpoints(publication):
        raise ValueError("incomplete behavior checkpoint inventory")
    _publish_rows(
        cfg,
        k,
        cfg.seat_behavior_summaries_path(k),
        Q7,
        (r for p in summaries for r in _rows(p, Q7, budget)),
        [publication, *summaries],
    )
    _publish_opportunity(cfg, d, opportunities, publication, budget)
    output_paths = behavior_paths(cfg, k)
    write_stage_done(
        stage_done_path(cfg.metrics_per_k_dir(k), "seat_behavior"),
        inputs=inputs,
        outputs=output_paths,
        cfg=cfg,
        stage="metrics",
        sidecar_artifacts=output_paths,
    )
    return output_paths


def _reconcile_all_player(cfg: AppConfig, d: _Design, budget: int) -> None:
    from itertools import groupby

    from farkle.analysis.all_player_metrics import all_player_batch_schema

    old = iter(_rows(cfg.metrics_all_player_batch_path(d.k), all_player_batch_schema(), budget))
    current = next(old, None)
    mapping = {
        "score_sum": "raw_final_score_sum",
        "turns_sum": "raw_n_turns_sum",
        "turn_round_mismatch_count": "raw_turn_round_mismatch_count",
        "score_per_turn_sum": "raw_turn_return_game_weighted_exact_sum",
        "score_per_round_sum": "raw_turn_return_round_proxy_sum",
        **{f"{n}_sum": f"raw_{n}_sum" for n in BEHAVIORS},
    }
    for key, group in groupby(
        _rows(cfg.seat_behavior_batch_path(d.k), B7, budget),
        key=lambda r: (r["deterministic_batch_id"], r["strategy"]),
    ):
        total = _zero()
        counts = {"completed": 0, "safety_limit": 0}
        for row in group:
            _add(total, row)
            counts[row["termination_status"]] += row["exposures"]
        if not total["exposures"]:
            continue
        if current is None or (
            current["root_seed"],
            current["k"],
            current["deterministic_batch_id"],
            current["strategy"],
        ) != (d.root, d.k, *key):
            raise ValueError("missing compatible all-player behavior support")
        if (
            current["raw_player_game_exposures"] != total["exposures"]
            or current["raw_completed_player_game_exposures"] != counts["completed"]
            or current["raw_safety_limit_player_game_exposures"] != counts["safety_limit"]
        ):
            raise ValueError("all-player exposure reconciliation failed")
        if any(
            not math.isclose(total[n], current[other], rel_tol=1e-12, abs_tol=1e-9)
            for n, other in mapping.items()
        ):
            raise ValueError("all-player behavior totals do not reconcile")
        current = next(old, None)
    if current is not None:
        raise ValueError("unexpected all-player support")


def _publish_opportunity(
    cfg: AppConfig, d: _Design, batches: list[dict[str, Any]], publication: Path, budget: int
) -> None:
    completed = _checked(sum(b["completed_games"] for b in batches))
    total_turns = _checked(sum(b["completed_total_turns"] for b in batches))
    turn_range = _checked(sum(b["completed_turn_range_sum"] for b in batches))
    seats = dict.fromkeys(range(1, d.k + 1), 0)
    for row in _rows(cfg.seat_behavior_batch_path(d.k), B7, budget):
        if row["termination_status"] == "completed":
            seats[row["seat"]] = _checked(seats[row["seat"]] + row["turns_sum"])
    if sum(seats.values()) != total_turns or (completed and total_turns <= 0):
        raise ValueError("game and exposure turn totals disagree")
    shares = {s: turns / total_turns if completed else None for s, turns in seats.items()}
    common = {
        "root_seed": d.root,
        "k": d.k,
        "completed_games": completed,
        "status": "ok" if completed else "no_completed_support",
    }
    rows = [
        {
            **common,
            "seat": 0,
            "turns_sum": total_turns,
            "mean_turns": None,
            "turn_share": None,
            "mean_turn_range": turn_range / completed if completed else None,
            "turn_share_imbalance": (
                0.5 * math.fsum(abs(seats[s] / total_turns - 1 / d.k) for s in seats)
                if completed
                else None
            ),
        }
    ]
    rows += [
        {
            **common,
            "seat": s,
            "turns_sum": turns,
            "mean_turns": turns / completed if completed else None,
            "turn_share": shares[s],
            "mean_turn_range": None,
            "turn_share_imbalance": None,
        }
        for s, turns in seats.items()
    ]
    _publish_rows(
        cfg,
        d.k,
        cfg.turn_opportunity_path(d.k),
        OPPORTUNITY_SCHEMA,
        iter(rows),
        [publication, cfg.turn_opportunity_batch_path(d.k), cfg.seat_behavior_batch_path(d.k)],
    )


def build_equal_k_opportunity(cfg: AppConfig) -> Path:
    ks = sorted(set(cfg.sim.n_players_list))
    if not ks:
        raise ValueError("opportunity requires configured k support")
    inputs = [cfg.turn_opportunity_path(k) for k in ks]
    output = cfg.turn_opportunity_equal_k_path()
    done = stage_done_path(output.parent, "turn_opportunity_equal_k")
    if stage_is_up_to_date(
        done, inputs=inputs, outputs=[output], cfg=cfg, stage="metrics", sidecar_artifacts=[output]
    ):
        return output
    values = []
    for k, path in zip(ks, inputs, strict=True):
        validate_artifact_sidecar(path, expected={"operation": "seat_behavior_v1"})
        rows = list(_rows(path, OPPORTUNITY_SCHEMA, cfg.resources.stage_batch_bytes["analysis"]))
        if len(rows) != k + 1 or [(r["root_seed"], r["k"], r["seat"]) for r in rows] != [
            (cfg.sim.seed, k, s) for s in range(k + 1)
        ]:
            raise ValueError("opportunity lacks required root/k/seat support")
        values.append(rows[0]["mean_turn_range"])
    row = {
        "root_seed": cfg.sim.seed,
        "required_ks": ks,
        "weights": [1 / len(ks)] * len(ks),
        "mean_turn_range_equal_k": (
            math.fsum(values) / len(values) if all(v is not None for v in values) else None
        ),
        "status": "ok" if all(v is not None for v in values) else "no_completed_support",
    }
    metadata = make_artifact_sidecar(
        cfg,
        output,
        producer="metrics",
        scope="across_k",
        source_scope="by_k",
        operation="turn_opportunity_v1",
        weighted_quantity="mean_game_turn_range",
        support_count_role="completed_games",
        uncertainty_method="none",
        replication_unit="game",
        conditioning="completed",
        source_artifacts=inputs,
        player_counts=ks,
        required_player_counts=ks,
        k_aggregation_method="equal_k",
        missing_cell_policy="fail",
        consistency_columns=OA.names,
    )
    write_parquet_artifact_atomic(
        pa.Table.from_pylist([row], schema=OA), output, sidecar=metadata, codec=cfg.parquet_codec
    )
    write_stage_done(
        done, inputs=inputs, outputs=[output], cfg=cfg, stage="metrics", sidecar_artifacts=[output]
    )
    return output
