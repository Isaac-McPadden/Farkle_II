"""Bounded matched finite-grid responses; observed contrasts, separate OOF means."""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from farkle.analysis.run_hgb import FEATURE_SPECS
from farkle.config import AppConfig
from farkle.simulation.strategies import STRATEGY_TUPLE_FIELDS, FavorDiceOrScore, ThresholdStrategy
from farkle.utils.artifact_contract import (
    make_artifact_sidecar,
    sha256_file,
    validate_artifact_sidecar,
)
from farkle.utils.authenticated_contract import identity_sha256
from farkle.utils.stage_completion import stage_done_path, stage_is_up_to_date, write_stage_done
from farkle.utils.writer import ParquetShardWriter

METHOD_VERSION = 1
PAIRS = (
    ("score_threshold", "dice_threshold"),
    ("auto_hot_dice", "run_up_score"),
    ("smart_five", "auto_hot_dice"),
)
POLICY = {
    "contract_id": "first_release_task0_v1",
    "method": "parameter_response_v1",
    "features": [n for n, _ in FEATURE_SPECS],
    "pairs": PAIRS,
    "max_levels": 25,
    "matching": "all_selected_levels_equal_background_weight",
    "target": "all_attempt_win_rate",
    "contrast": "observed_11-observed_10-observed_01+observed_00",
}


def response_schema(product: str) -> pa.Schema:
    fields: list[pa.Field] = [
        pa.field("root_seed", pa.int64(), False),
        pa.field("k", pa.int16(), False),
    ]
    if product == "one_way":
        fields += [
            pa.field("parameter", pa.string(), False),
            pa.field("level", pa.int32(), False),
            pa.field("matched_backgrounds", pa.int64(), False),
        ]
        fields += [pa.field(n, pa.float64()) for n in ("observed_mean", "oof_predicted_mean")]
    elif product == "two_way":
        fields += [pa.field(n, pa.string(), False) for n in ("parameter_a", "parameter_b")]
        fields += [pa.field(n, pa.int32(), False) for n in ("a0", "a1", "b0", "b1")]
        fields += [pa.field("matched_backgrounds", pa.int64(), False)]
        fields += [
            pa.field(n, pa.float64())
            for n in (
                "observed_00",
                "observed_10",
                "observed_01",
                "observed_11",
                "observed_nonadditivity",
                "oof_mean_00",
                "oof_mean_10",
                "oof_mean_01",
                "oof_mean_11",
            )
        ]
    elif product == "support":
        return pa.schema(
            fields
            + [
                pa.field("support_id", pa.string(), False),
                pa.field("product", pa.string(), False),
                pa.field("specification_json", pa.string(), False),
                pa.field("background_id", pa.string()),
                pa.field("corner_or_level", pa.string()),
                pa.field("strategy", pa.int32()),
                pa.field("fold", pa.int16()),
                pa.field("status", pa.string(), False),
            ]
        )
    else:
        raise ValueError("unknown response product")
    return pa.schema(
        fields
        + [pa.field("support_id", pa.string(), False), pa.field("status", pa.string(), False)]
    )


def selected_levels(values: list[int]) -> list[int]:
    levels = sorted(set(values))
    return levels if len(levels) <= 25 else [levels[i * (len(levels) - 1) // 24] for i in range(25)]


def _canonical(row: dict[str, Any]) -> dict[str, Any]:
    attrs = {n: row[n] for n in STRATEGY_TUPLE_FIELDS}
    for n, v in attrs.items():
        if n in ("score_threshold", "dice_threshold"):
            if type(v) is not int or not -(2**31) <= v < 2**31:
                raise ValueError(f"invalid canonical threshold {n}")
        elif n == "favor_dice_or_score":
            attrs[n] = FavorDiceOrScore(v).value
        elif type(v) is not bool:
            raise ValueError(f"invalid canonical boolean {n}")
    ThresholdStrategy(
        **{**attrs, "favor_dice_or_score": FavorDiceOrScore(attrs["favor_dice_or_score"])}
    )
    # Ignore inactive numerical sentinels when identifying equivalent settings.
    for name, flag in (("score_threshold", "consider_score"), ("dice_threshold", "consider_dice")):
        if not attrs[flag]:
            attrs[name] = None
    return attrs


def response_rows(
    manifest: list[dict[str, Any]],
    performance: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    folds: list[dict[str, Any]],
    *,
    root: int,
    k: int,
    source_identity: dict[str, str],
    prediction_unavailable: bool = False,
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Enumerate one matched specification at a time, with exact support attributes."""
    roster = {r["strategy_id"]: r for r in manifest}
    observed = {r["strategy"]: r for r in performance}
    predicted = {r["strategy"]: r for r in predictions}
    if (
        len(roster) != len(manifest)
        or len(observed) != len(performance)
        or set(roster) != set(observed)
    ):
        raise ValueError("response requires unique complete roster/performance support")
    if (
        len(predicted) != len(predictions)
        or (not prediction_unavailable and set(predicted) != set(roster))
        or (prediction_unavailable and predictions)
    ):
        raise ValueError("response requires all promised held-out predictions")
    fold_map = {r["fold"]: r for r in folds}
    if len(fold_map) != len(folds):
        raise ValueError("duplicate HGB fold")
    for sid, row in observed.items():
        if (
            row.get("root_seed", root) != root
            or row.get("k", row.get("n_players", k)) != k
            or not isinstance(row["win_rate"], (int, float))
            or not math.isfinite(row["win_rate"])
            or not 0 <= row["win_rate"] <= 1
        ):
            raise ValueError("invalid response performance root/k/target")
        if not prediction_unavailable:
            pred = predicted[sid]
            fold = pred["fold"]
            if (
                type(fold) is not int
                or not 0 <= fold < 32768
                or fold not in fold_map
                or pred["root_seed"] != root
                or pred["players"] != k
                or not math.isfinite(pred["predicted_win_rate"])
                or not math.isclose(pred["observed_win_rate"], row["win_rate"], abs_tol=1e-7)
            ):
                raise ValueError("invalid held-out prediction/fold/target binding")
    for fold, row in fold_map.items():
        if (
            row["root_seed"] != root
            or row["players"] != k
            or row["heldout_strategies"] != sum(p["fold"] == fold for p in predictions)
            or row["train_strategies"] + row["heldout_strategies"] != len(roster)
        ):
            raise ValueError("HGB fold support does not reconcile")
    attrs: dict[int, dict[str, Any]] = {}
    excluded: dict[str, list[int]] = {}
    seen: dict[str, int] = {}
    for sid, row in sorted(roster.items()):
        kind = row.get("strategy_type", "ThresholdStrategy")
        if kind != "ThresholdStrategy":
            excluded.setdefault(kind, []).append(sid)
            continue
        value = _canonical(row)
        key = json.dumps(value, sort_keys=True)
        if key in seen:
            raise ValueError("equivalent canonical settings have distinct strategy IDs")
        seen[key] = sid
        attrs[sid] = value
    base = {"root_seed": root, "k": k}

    def support(
        spec: dict[str, Any],
        matched: list[tuple[str, dict[str, Any], dict[tuple[int, ...], int]]],
        product: str,
        corners: list[tuple[int, ...]],
        labels: list[str],
    ):
        binding = {
            "policy": POLICY,
            "sources": source_identity,
            "specification": spec,
            "matched": [
                [background, [[list(c), ids[c]] for c in corners]] for background, _, ids in matched
            ],
        }
        support_id = identity_sha256(binding)
        status = "ok" if matched else "unestimable_common_support"
        if matched and prediction_unavailable:
            status = "prediction_unavailable"
        records = []
        if not matched:
            records.append(
                {
                    **base,
                    "support_id": support_id,
                    "product": product,
                    "specification_json": json.dumps(spec, sort_keys=True, separators=(",", ":")),
                    "background_id": None,
                    "corner_or_level": None,
                    "strategy": None,
                    "fold": None,
                    "status": status,
                }
            )

        # Returned generator keeps the matched-ID inventory out of memory.
        def inventory():
            yield from records
            for background, attributes, ids in matched:
                exact = {**spec, "matching_attributes": attributes}
                for corner, label in zip(corners, labels, strict=True):
                    sid = ids[corner]
                    yield {
                        **base,
                        "support_id": support_id,
                        "product": product,
                        "specification_json": json.dumps(
                            exact, sort_keys=True, separators=(",", ":")
                        ),
                        "background_id": background,
                        "corner_or_level": label,
                        "strategy": sid,
                        "fold": None if prediction_unavailable else predicted[sid]["fold"],
                        "status": status,
                    }

        return support_id, status, inventory()

    for kind, ids in sorted(excluded.items()):
        spec = {"strategy_type": kind, "excluded_strategy_ids": ids, "excluded_count": len(ids)}
        yield "support", {
            **base,
            "support_id": identity_sha256({"policy": POLICY, "sources": source_identity, **spec}),
            "product": "excluded_type",
            "specification_json": json.dumps(spec, sort_keys=True, separators=(",", ":")),
            "background_id": None,
            "corner_or_level": None,
            "strategy": None,
            "fold": None,
            "status": "unsupported_type",
        }
    for parameters in [*((n,) for n, _ in FEATURE_SPECS), *PAIRS]:
        canonical = tuple("favor_dice_or_score" if n == "favor_score" else n for n in parameters)
        eligible = {sid: a for sid, a in attrs.items() if all(a[n] is not None for n in canonical)}

        def level(a: dict[str, Any], n: str) -> int:
            return (
                int(a[n] == FavorDiceOrScore.SCORE.value)
                if n == "favor_dice_or_score"
                else int(a[n])
            )

        levels = [selected_levels([level(a, n) for a in eligible.values()]) for n in canonical]
        groups: dict[str, tuple[dict[str, Any], dict[tuple[int, ...], int]]] = {}
        for sid, a in eligible.items():
            coordinates = tuple(level(a, n) for n in canonical)
            background = {n: v for n, v in a.items() if n not in canonical}
            key = json.dumps(background, sort_keys=True, separators=(",", ":"))
            groups.setdefault(key, (background, {}))[1][coordinates] = sid
        product = "one_way" if len(parameters) == 1 else "two_way"
        rectangles: list[Any] = (
            [None]
            if len(parameters) == 1
            else list(
                itertools.product(
                    zip(levels[0], levels[0][1:], strict=False),
                    zip(levels[1], levels[1][1:], strict=False),
                )
            )
        )
        if not rectangles or not all(levels):
            spec = {
                "parameters": parameters,
                "selected_levels": levels,
                "reason": "insufficient_active_levels",
            }
            _, _, records = support(spec, [], product, [], [])
            for record in records:
                yield "support", record
            continue
        for rectangle in rectangles:
            corners: list[tuple[int, ...]] = (
                [(v,) for v in levels[0]]
                if rectangle is None
                else [
                    (rectangle[0][0], rectangle[1][0]),
                    (rectangle[0][1], rectangle[1][0]),
                    (rectangle[0][0], rectangle[1][1]),
                    (rectangle[0][1], rectangle[1][1]),
                ]
            )
            labels = [str(v[0]) for v in corners] if rectangle is None else ["00", "10", "01", "11"]
            matched = [
                (identity_sha256(a), a, ids)
                for _, (a, ids) in sorted(groups.items())
                if all(c in ids for c in corners)
            ]
            spec = {
                "parameters": parameters,
                "selected_levels": levels,
                "corners": corners,
                "weighting": "equal_matched_backgrounds",
            }
            support_id, status, records = support(spec, matched, product, corners, labels)
            for record in records:
                yield "support", record
            obs = [
                (
                    math.fsum(observed[ids[c]]["win_rate"] for _, _, ids in matched) / len(matched)
                    if matched
                    else None
                )
                for c in corners
            ]
            oof = [
                (
                    math.fsum(predicted[ids[c]]["predicted_win_rate"] for _, _, ids in matched)
                    / len(matched)
                    if matched and not prediction_unavailable
                    else None
                )
                for c in corners
            ]
            common = {
                **base,
                "matched_backgrounds": len(matched),
                "support_id": support_id,
                "status": status,
            }
            if rectangle is None:
                for c, mean, prediction_mean in zip(corners, obs, oof, strict=True):
                    yield product, {
                        **common,
                        "parameter": parameters[0],
                        "level": c[0],
                        "observed_mean": mean,
                        "oof_predicted_mean": prediction_mean,
                    }
            else:
                contrast = (
                    math.fsum(
                        observed[ids[corners[3]]]["win_rate"]
                        - observed[ids[corners[1]]]["win_rate"]
                        - observed[ids[corners[2]]]["win_rate"]
                        + observed[ids[corners[0]]]["win_rate"]
                        for _, _, ids in matched
                    )
                    / len(matched)
                    if matched
                    else None
                )
                yield product, {
                    **common,
                    "parameter_a": parameters[0],
                    "parameter_b": parameters[1],
                    "a0": rectangle[0][0],
                    "a1": rectangle[0][1],
                    "b0": rectangle[1][0],
                    "b1": rectangle[1][1],
                    **dict(zip([f"observed_{s}" for s in labels], obs, strict=True)),
                    **dict(zip([f"oof_mean_{s}" for s in labels], oof, strict=True)),
                    "observed_nonadditivity": contrast,
                }


def response_paths(cfg: AppConfig, k: int) -> tuple[Path, ...]:
    return (
        cfg.parameter_responses_path(k),
        cfg.parameter_nonadditivity_path(k),
        cfg.parameter_response_support_path(k),
    )


def build_parameter_responses(cfg: AppConfig, k: int, *, force: bool = False) -> tuple[Path, ...]:
    from contextlib import ExitStack

    inputs = [
        cfg.strategy_manifest_root_path(),
        cfg.performance_by_k_path(k),
        cfg.hgb_predictive_scores_path(k),
        cfg.hgb_fold_metrics_path(k),
    ]
    paths = response_paths(cfg, k)
    done = stage_done_path(cfg.hgb_per_k_dir(k), "parameter_responses")
    if not force and stage_is_up_to_date(
        done, inputs=inputs, outputs=paths, cfg=cfg, stage="hgb", sidecar_artifacts=paths
    ):
        return paths
    budget = cfg.resources.stage_batch_bytes["analysis"]
    data = []
    projections = [
        ["strategy_id", *STRATEGY_TUPLE_FIELDS],
        ["strategy", "win_rate"],
        ["strategy", "root_seed", "players", "fold", "observed_win_rate", "predicted_win_rate"],
        ["root_seed", "players", "fold", "train_strategies", "heldout_strategies"],
    ]
    admitted = 0
    for index, path in enumerate(inputs):
        validate_artifact_sidecar(path)
        parquet = pq.ParquetFile(path)
        admitted += parquet.metadata.num_rows * 1536
        if admitted > budget:
            raise MemoryError("parameter response roster exceeds admitted compact-grid budget")
        columns = projections[index] + [
            n
            for n in ("strategy_type", "root_seed", "k", "n_players")
            if n in parquet.schema_arrow.names and n not in projections[index]
        ]
        data.append(parquet.read(columns=columns).to_pylist())
    source_identity = {str(p): sha256_file(p) for p in inputs}
    with ExitStack() as stack:
        writers = {}
        buffers: dict[str, list[dict[str, Any]]] = {}
        for product, path in zip(("one_way", "two_way", "support"), paths, strict=True):
            schema = response_schema(product)
            metadata = make_artifact_sidecar(
                cfg,
                path,
                producer="hgb",
                scope="by_k",
                source_scope="by_k_and_single_root",
                operation="parameter_response_v1",
                weighted_quantity="all_attempt_win_probability",
                support_count_role="matched_canonical_strategy_backgrounds",
                uncertainty_method="none_exploratory",
                replication_unit="strategy_configuration",
                conditioning="finite_grid_descriptive_predictive_association_not_causal",
                source_artifacts=inputs,
                player_counts=[k],
                required_player_counts=[k],
                missing_cell_policy="fail",
                consistency_columns=schema.names,
                method_contract={
                    "kind": "operation",
                    "procedure": "parameter_response_v1",
                    "parameters": POLICY,
                },
            )
            writers[product] = stack.enter_context(
                ParquetShardWriter(
                    str(path), schema=schema, sidecar=metadata, temp_dir=cfg.io.temp_dir
                )
            )
            buffers[product] = []
        for product, row in response_rows(
            *data, root=cfg.sim.seed, k=k, source_identity=source_identity
        ):
            buffers[product].append(row)
            if len(buffers[product]) >= max(1, budget // (16 * 8192)):
                writers[product].write_batch(
                    pa.Table.from_pylist(buffers[product], schema=response_schema(product))
                )
                buffers[product].clear()
        for product, writer in writers.items():
            writer.write_batch(
                pa.Table.from_pylist(buffers[product], schema=response_schema(product))
            )
    write_stage_done(
        done, inputs=inputs, outputs=paths, cfg=cfg, stage="hgb", sidecar_artifacts=paths
    )
    return paths
