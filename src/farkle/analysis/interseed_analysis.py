# src/farkle/analysis/interseed_analysis.py
"""Run only the cross-seed analysis stages and record a summary."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import pyarrow as pa
from scipy.stats import kendalltau, spearmanr

from farkle.analysis import stage_logger
from farkle.analysis.game_stats_interseed import SeedInputs, _seed_analysis_dirs
from farkle.analysis.stage_registry import resolve_interseed_stage_layout
from farkle.analysis.stage_state import stage_done_path, stage_is_up_to_date, write_stage_done
from farkle.config import AppConfig
from farkle.utils.artifacts import write_parquet_atomic
from farkle.utils.writer import atomic_path

LOGGER = logging.getLogger(__name__)

SUMMARY_NAME = "interseed_summary.json"


def run(
    cfg: AppConfig,
    *,
    force: bool = False,
    run_stages: bool = True,
    run_rng_diagnostics: bool | None = None,
) -> None:
    """Execute or summarize rng/variance/meta/TrueSkill/agreement interseed analytics."""

    previous_layout = cfg._stage_layout
    cfg.set_stage_layout(resolve_interseed_stage_layout(cfg))

    stage_log = stage_logger("interseed", logger=LOGGER)
    stage_log.start()

    try:
        interseed_ready, interseed_reason = cfg.interseed_ready()
        if not interseed_ready:
            stage_log.missing_input(interseed_reason)
            return

        statuses: dict[str, dict[str, Any]] = {}

        rng_enabled = (
            run_rng_diagnostics
            if run_rng_diagnostics is not None
            else not cfg.analysis.disable_rng_diagnostics
        )
        variance_enabled = True
        game_stats_enabled = True
        meta_enabled = True
        trueskill_enabled = True
        agreement_enabled = True
        s_tier_stability_enabled = True

        if rng_enabled and run_stages:
            from farkle.analysis import rng_diagnostics

            rng_diagnostics.run(cfg, force=force)
        statuses["rng_diagnostics"] = {
            "enabled": rng_enabled,
            "outputs": _existing_paths(_rng_outputs(cfg)) if rng_enabled else [],
        }

        if variance_enabled and run_stages:
            from farkle.analysis import variance

            variance.run(cfg, force=force)
        statuses["variance"] = {
            "enabled": variance_enabled,
            "outputs": _existing_paths(_variance_outputs(cfg)),
        }

        if game_stats_enabled and run_stages:
            from farkle.analysis import game_stats_interseed

            game_stats_interseed.run(cfg, force=force)
        statuses["game_stats_interseed"] = {
            "enabled": game_stats_enabled,
            "outputs": _existing_paths(_game_stats_outputs(cfg)),
        }

        if meta_enabled and run_stages:
            from farkle.analysis import meta

            meta.run(cfg, force=force)
        statuses["meta"] = {
            "enabled": meta_enabled,
            "outputs": _existing_paths(_meta_outputs(cfg)),
        }

        if trueskill_enabled and run_stages:
            from farkle.analysis import trueskill

            trueskill.run(cfg)
        statuses["trueskill"] = {
            "enabled": trueskill_enabled,
            "outputs": _existing_paths(_trueskill_outputs(cfg)),
        }

        if agreement_enabled and run_stages:
            from farkle.analysis import agreement

            agreement.run(cfg)
        statuses["agreement"] = {
            "enabled": agreement_enabled,
            "outputs": _existing_paths(_agreement_outputs(cfg)),
        }

        if s_tier_stability_enabled and run_stages:
            _run_s_tier_stability(cfg, force=force)
        statuses["s_tier_stability"] = {
            "enabled": s_tier_stability_enabled,
            "outputs": _existing_paths(_s_tier_stability_outputs(cfg)),
        }

        summary_path = cfg.interseed_stage_dir / SUMMARY_NAME
        done_path = stage_done_path(cfg.interseed_stage_dir, "interseed")
        inputs = sorted({Path(path) for path in _flatten_output_paths(statuses)})

        if not force and stage_is_up_to_date(
            done_path,
            inputs=inputs,
            outputs=[summary_path],
            config_sha=cfg.config_sha,
        ):
            LOGGER.info(
                "Interseed summary up-to-date",
                extra={"stage": "interseed", "path": str(summary_path)},
            )
            return

        payload = {
            "config_sha": cfg.config_sha,
            "run_interseed": cfg.analysis.run_interseed,
            "interseed_ready": interseed_ready,
            "stages": statuses,
        }
        with atomic_path(str(summary_path)) as tmp_path:
            Path(tmp_path).write_text(json.dumps(payload, indent=2, sort_keys=True))

        write_stage_done(
            done_path,
            inputs=inputs,
            outputs=[summary_path],
            config_sha=cfg.config_sha,
        )

        LOGGER.info(
            "Interseed summary written",
            extra={"stage": "interseed", "path": str(summary_path)},
        )
    finally:
        cfg._stage_layout = previous_layout


def _existing_paths(paths: list[Path]) -> list[str]:
    return [str(path) for path in paths if path.exists()]


def _flatten_output_paths(statuses: Mapping[str, Mapping[str, Any]]) -> list[str]:
    outputs: list[str] = []
    for details in statuses.values():
        outputs.extend(details.get("outputs", []))
    return outputs


def _variance_outputs(cfg: AppConfig) -> list[Path]:
    return [
        cfg.variance_output_path("variance.parquet"),
        cfg.variance_output_path("variance_summary.parquet"),
        cfg.variance_output_path("variance_components.parquet"),
    ]


def _meta_outputs(cfg: AppConfig) -> list[Path]:
    outputs: list[Path] = []
    for players in sorted({int(n) for n in cfg.sim.n_players_list}):
        outputs.append(cfg.meta_output_path(players, f"strategy_summary_{players}p_meta.parquet"))
        outputs.append(cfg.meta_output_path(players, f"meta_{players}p.json"))
    return outputs


def _game_stats_outputs(cfg: AppConfig) -> list[Path]:
    return [
        cfg.interseed_stage_dir / "game_length_interseed.parquet",
        cfg.interseed_stage_dir / "margin_interseed.parquet",
    ]


def _trueskill_outputs(cfg: AppConfig) -> list[Path]:
    outputs = [
        cfg.trueskill_path("ratings_pooled.parquet"),
        cfg.trueskill_path("ratings_pooled.json"),
        cfg.trueskill_stage_dir / "tiers.json",
    ]
    outputs.extend(sorted(cfg.trueskill_pooled_dir.glob("ratings_pooled_seed*.parquet")))
    outputs.extend(sorted(cfg.trueskill_stage_dir.glob("ratings_pooled_seed*.parquet")))
    return outputs


def _agreement_outputs(cfg: AppConfig) -> list[Path]:
    outputs = [cfg.agreement_output_path(players) for players in cfg.agreement_players()]
    if cfg.agreement_include_pooled():
        outputs.append(cfg.agreement_output_path_pooled())
    outputs.append(cfg.agreement_stage_dir / "agreement_summary.parquet")
    return outputs


def _rng_outputs(cfg: AppConfig) -> list[Path]:
    return [cfg.rng_output_path("rng_diagnostics.parquet")]


def _s_tier_stability_outputs(cfg: AppConfig) -> list[Path]:
    output_dir = cfg.interseed_stage_dir
    return [output_dir / "s_tier_stability.json", output_dir / "s_tier_stability.parquet"]


@dataclass(frozen=True)
class SeedTierData:
    seed: int
    analysis_dir: Path
    s_tiers: dict[str, str]
    s_tiers_source: str
    s_tiers_path: Path | None
    ranking: dict[str, int] | None
    ranking_path: Path | None
    input_paths: list[Path]


def _run_s_tier_stability(cfg: AppConfig, *, force: bool = False) -> None:
    stage_log = stage_logger("s_tier_stability", logger=LOGGER)
    stage_log.start()

    seeds = _seed_analysis_dirs(cfg)
    if not seeds:
        stage_log.missing_input("no seed analysis directories resolved")
        return

    seed_data: list[SeedTierData] = []
    for seed_entry in seeds:
        data = _load_seed_tier_data(cfg, seed_entry)
        if data is None:
            continue
        seed_data.append(data)

    if len(seed_data) < 2:
        stage_log.missing_input("fewer than two seeds with S-tier data")
        return

    output_dir = cfg.interseed_stage_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    json_output = output_dir / "s_tier_stability.json"
    parquet_output = output_dir / "s_tier_stability.parquet"
    done_path = stage_done_path(output_dir, "interseed.s_tier_stability")

    input_paths: list[Path] = []
    for data in seed_data:
        input_paths.extend(data.input_paths)
    inputs = sorted({path for path in input_paths if path.exists()})
    outputs = [json_output, parquet_output]

    if inputs and not force and stage_is_up_to_date(
        done_path,
        inputs=inputs,
        outputs=outputs,
        config_sha=cfg.config_sha,
    ):
        LOGGER.info(
            "S-tier stability outputs up-to-date",
            extra={"stage": "s_tier_stability", "path": str(done_path)},
        )
        return

    tier_rows: list[dict[str, object]] = []
    pairs_payload: list[dict[str, object]] = []
    tier_labels = _order_tier_labels(
        {label for data in seed_data for label in data.s_tiers.values()}
    )

    seed_lookup = {data.seed: data for data in seed_data}
    for seed_a, seed_b in combinations(sorted(seed_lookup), 2):
        data_a = seed_lookup[seed_a]
        data_b = seed_lookup[seed_b]

        tier_sets_a = _tiers_to_sets(data_a.s_tiers)
        tier_sets_b = _tiers_to_sets(data_b.s_tiers)
        jaccard_by_tier = {
            label: _jaccard_index(tier_sets_a.get(label, set()), tier_sets_b.get(label, set()))
            for label in tier_labels
        }

        rank_corr = _rank_correlations(data_a.ranking, data_b.ranking)

        flip_rows, flip_summary = _tier_flips(data_a, data_b)
        tier_rows.extend(flip_rows)

        pairs_payload.append(
            {
                "seed_a": seed_a,
                "seed_b": seed_b,
                "tier_labels": tier_labels,
                "jaccard_by_tier": jaccard_by_tier,
                "rank_correlations": rank_corr,
                "tier_flips": flip_summary,
            }
        )

    payload = {
        "config_sha": cfg.config_sha,
        "seeds": sorted(seed_lookup),
        "tier_labels": tier_labels,
        "seed_summaries": {
            str(data.seed): {
                "s_tiers_source": data.s_tiers_source,
                "s_tiers_path": str(data.s_tiers_path) if data.s_tiers_path else None,
                "ranking_path": str(data.ranking_path) if data.ranking_path else None,
                "tier_counts": _tier_counts(data.s_tiers),
                "strategy_count": len(data.s_tiers),
            }
            for data in seed_data
        },
        "pairs": pairs_payload,
    }

    with atomic_path(str(json_output)) as tmp_path:
        Path(tmp_path).write_text(json.dumps(payload, indent=2, sort_keys=True))

    tier_frame = pd.DataFrame(tier_rows)
    if tier_frame.empty:
        table = pa.Table.from_pylist([], schema=_tier_flip_schema())
    else:
        table = pa.Table.from_pandas(tier_frame, preserve_index=False)
    write_parquet_atomic(table, parquet_output, codec=cfg.parquet_codec)

    write_stage_done(
        done_path,
        inputs=inputs,
        outputs=outputs,
        config_sha=cfg.config_sha,
    )


def _load_seed_tier_data(cfg: AppConfig, seed_input: SeedInputs) -> SeedTierData | None:
    analysis_dir = seed_input.analysis_dir
    s_tiers_path = _resolve_h2h_path(cfg, analysis_dir, "h2h_s_tiers.json")
    ranking_path = _resolve_h2h_path(cfg, analysis_dir, "h2h_significant_ranking.csv")

    input_paths: list[Path] = []

    s_tiers: dict[str, str] = {}
    s_tiers_source = "missing"
    if s_tiers_path is not None and s_tiers_path.exists():
        s_tiers = _load_s_tiers(s_tiers_path)
        if s_tiers:
            s_tiers_source = "h2h_s_tiers.json"
            input_paths.append(s_tiers_path)

    ranking: dict[str, int] | None = None
    ranking_order: list[str] = []
    if ranking_path is not None and ranking_path.exists():
        ranking_order, ranking = _load_ranking(ranking_path)
        if ranking:
            input_paths.append(ranking_path)

    if not s_tiers and ranking_order:
        union_candidates, union_path = _load_union_candidates(cfg, analysis_dir)
        if union_candidates:
            ranking_order = [name for name in ranking_order if name in union_candidates]
            if union_path is not None:
                input_paths.append(union_path)
        s_tiers = _assign_s_tiers(ranking_order)
        if s_tiers:
            s_tiers_source = "ranking"

    if not s_tiers:
        return None

    return SeedTierData(
        seed=seed_input.seed,
        analysis_dir=analysis_dir,
        s_tiers=s_tiers,
        s_tiers_source=s_tiers_source,
        s_tiers_path=s_tiers_path if s_tiers_source == "h2h_s_tiers.json" else None,
        ranking=ranking,
        ranking_path=ranking_path if ranking else None,
        input_paths=input_paths,
    )


def _resolve_h2h_path(cfg: AppConfig, analysis_dir: Path, filename: str) -> Path | None:
    candidates: list[Path] = []
    for stage in ("post_h2h", "head2head"):
        folder = cfg._interseed_input_folder(stage)
        if folder is not None:
            candidates.append(analysis_dir / folder / filename)
        candidates.extend(sorted(analysis_dir.glob(f"*_{stage}"))[:1])
    candidates.append(analysis_dir / filename)
    for candidate in candidates:
        if candidate.is_dir():
            candidate = candidate / filename
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else None


def _load_s_tiers(path: Path) -> dict[str, str]:
    try:
        payload_any: Any = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload_any, Mapping):
        return {}

    validated: dict[str, str] = {}
    for key, value in payload_any.items():
        if key == "_meta":
            continue
        if not isinstance(value, str):
            continue
        validated[str(key)] = value
    return validated


def _load_ranking(path: Path) -> tuple[list[str], dict[str, int]]:
    df = pd.read_csv(path)
    if "strategy" not in df.columns:
        return [], {}
    df = df.copy()
    df["strategy"] = df["strategy"].astype(str)
    if "rank" in df.columns:
        df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
        df = df.dropna(subset=["rank"]).sort_values(["rank", "strategy"], kind="mergesort")
        order = df["strategy"].tolist()
    else:
        order = df["strategy"].tolist()
    ranking = {strategy: idx for idx, strategy in enumerate(order, start=1)}
    return order, ranking


def _load_union_candidates(
    cfg: AppConfig, analysis_dir: Path
) -> tuple[set[str], Path | None]:
    candidates: list[Path] = []
    folder = cfg._interseed_input_folder("head2head")
    if folder is not None:
        candidates.append(analysis_dir / folder / "h2h_union_candidates.json")
    candidates.extend(sorted(analysis_dir.glob("*_head2head"))[:1])
    candidates.append(analysis_dir / "h2h_union_candidates.json")
    for candidate in candidates:
        path = candidate
        if path.is_dir():
            path = path / "h2h_union_candidates.json"
        if not path.exists():
            continue
        try:
            payload_any: Any = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        raw_candidates: Any
        if isinstance(payload_any, dict) and "candidates" in payload_any:
            raw_candidates = payload_any.get("candidates")
        else:
            raw_candidates = payload_any
        if isinstance(raw_candidates, list):
            validated_candidates = {str(item) for item in raw_candidates}
            return validated_candidates, path
    return set(), None


def _assign_s_tiers(ordered: list[str]) -> dict[str, str]:
    tiers: dict[str, str] = {}
    for idx, strategy in enumerate(ordered):
        if idx < 10:
            label = "S+"
        elif idx < 30:
            label = "S"
        else:
            label = "S-"
        tiers[str(strategy)] = label
    return tiers


def _order_tier_labels(labels: Iterable[str]) -> list[str]:
    preferred = ["S+", "S", "S-"]
    label_set = {label for label in labels if label}
    ordered = [label for label in preferred if label in label_set]
    ordered.extend(sorted(label_set - set(preferred)))
    return ordered


def _tiers_to_sets(tiers: dict[str, str]) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = {}
    for strategy, label in tiers.items():
        grouped.setdefault(label, set()).add(strategy)
    return grouped


def _jaccard_index(set_a: set[str], set_b: set[str]) -> float | None:
    union = set_a | set_b
    if not union:
        return None
    return len(set_a & set_b) / len(union)


def _rank_correlations(
    ranks_a: Mapping[str, int] | None, ranks_b: Mapping[str, int] | None
) -> dict[str, object]:
    if not ranks_a or not ranks_b:
        return {
            "n_common": 0,
            "n_ranked_a": len(ranks_a or {}),
            "n_ranked_b": len(ranks_b or {}),
            "spearman": None,
            "kendall": None,
        }
    common = sorted(set(ranks_a) & set(ranks_b))
    n_common = len(common)
    if n_common < 2:
        return {
            "n_common": n_common,
            "n_ranked_a": len(ranks_a),
            "n_ranked_b": len(ranks_b),
            "spearman": None,
            "kendall": None,
        }
    vec_a = np.array([ranks_a[name] for name in common], dtype=float)
    vec_b = np.array([ranks_b[name] for name in common], dtype=float)
    spearman = spearmanr(vec_a, vec_b)
    kendall = kendalltau(vec_a, vec_b)
    spearman_payload = (
        None
        if np.isnan(spearman.statistic)
        else {"rho": float(spearman.statistic), "pvalue": float(spearman.pvalue)}
    )
    kendall_payload = (
        None
        if np.isnan(kendall.statistic)
        else {"tau": float(kendall.statistic), "pvalue": float(kendall.pvalue)}
    )
    return {
        "n_common": n_common,
        "n_ranked_a": len(ranks_a),
        "n_ranked_b": len(ranks_b),
        "spearman": spearman_payload,
        "kendall": kendall_payload,
    }


def _tier_flips(
    data_a: SeedTierData, data_b: SeedTierData
) -> tuple[list[dict[str, object]], dict[str, object]]:
    strategies = sorted(set(data_a.s_tiers) | set(data_b.s_tiers))
    rows: list[dict[str, object]] = []
    n_shared = 0
    n_flipped = 0
    for strategy in strategies:
        tier_a = data_a.s_tiers.get(strategy)
        tier_b = data_b.s_tiers.get(strategy)
        both_present = tier_a is not None and tier_b is not None
        flipped = bool(both_present and tier_a != tier_b)
        if both_present:
            n_shared += 1
            if flipped:
                n_flipped += 1
        rows.append(
            {
                "seed_a": data_a.seed,
                "seed_b": data_b.seed,
                "strategy_id": strategy,
                "tier_a": tier_a,
                "tier_b": tier_b,
                "present_in_both": both_present,
                "flipped": flipped if both_present else None,
            }
        )
    flip_rate = (n_flipped / n_shared) if n_shared else None
    summary: dict[str, object] = {
        "n_shared": n_shared,
        "n_flipped": n_flipped,
        "flip_rate": flip_rate,
    }
    return rows, summary


def _tier_counts(tiers: Mapping[str, str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in tiers.values():
        counts[label] = counts.get(label, 0) + 1
    return counts


def _tier_flip_schema() -> pa.Schema:
    schema_fields: list[tuple[str, pa.DataType]] = [
        ("seed_a", pa.int64()),
        ("seed_b", pa.int64()),
        ("strategy_id", pa.string()),
        ("tier_a", pa.string()),
        ("tier_b", pa.string()),
        ("present_in_both", pa.bool_()),
        ("flipped", pa.bool_()),
    ]
    return pa.schema(schema_fields)
