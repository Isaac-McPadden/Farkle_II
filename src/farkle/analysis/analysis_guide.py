"""Task 8: bounded evidence extraction and JSON-only guide rendering.

This consumer adds no statistical analysis, model fit, or random draws. Large
histograms and batch/support inventories remain in their owning artifacts.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import pyarrow.compute as pc
import pyarrow.parquet as pq

from farkle.analysis.guide_questions import QUESTIONS
from farkle.config import AppConfig
from farkle.orchestration.seed_utils import base_results_dir, resolve_results_dir
from farkle.utils.arrow_batches import iter_parquet_tables_by_bytes
from farkle.utils.artifact_contract import (
    ArtifactSidecar,
    read_json_file_with_retry,
    sha256_file,
    sidecar_path,
    validate_artifact_sidecar,
)
from farkle.utils.authenticated_contract import (
    CodeIdentityPolicy,
    load_authenticated_sidecar,
    resolve_code_identity,
    stage_config_identity,
)
from farkle.utils.release_identity import capture_v3_inputs

CONTRACT_ID = "first_release_task0_v1"
SCHEMA_VERSION = 1
SAMPLE_ROWS = 6

GROUPS = (
    (1, "Strategy performance"),
    (9, "Performance across player counts"),
    (18, "Risk and outcome distributions"),
    (26, "Opponent and matchup composition"),
    (36, "Head-to-head superiority and dominance"),
    (50, "Seat and turn-order effects"),
    (65, "Gameplay behavior and efficiency"),
    (73, "Randomness and reproducibility"),
    (85, "Explaining why strategies perform well"),
    (95, "Controls, validity, and claim boundaries"),
)
FRAMING = {
    "A34": "Tournament results characterize the configured opponent population; H2H answers direct finalist-versus-finalist questions.",
    "A38": "H2H inference is restricted to the frozen finalist family.",
    "A59": "Three distinct scoring-rate views retain their denominators.",
    "A83": "Two fixed roots are a reproducibility challenge, not a sample of a hypothetical population of seeds.",
    "A93": "HGB describes predictive association and candidate generation; new candidates require independent simulation.",
}
LIMITATIONS = [
    "Broad opponent-composition sensitivity and favorable-matchup mechanisms are deferred.",
    "Fixed-matchup winner entropy/probabilities are deferred; lag checks do not answer them.",
    "State-level decision analysis, dynamic strategies and causal risk/survival mechanisms are excluded.",
    "Prescribed strong/weak strategy-performance controls are excluded; integrity and rule-derived checks remain.",
    "All findings condition on the configured finite strategy population, rules and design.",
    "Outcome spread, multimodality and quantiles differ from uncertainty in a win-probability estimate.",
    "TrueSkill disagreement does not identify opponent-composition effects: model assumptions, update order and completed-game conditioning differ; sigma is not an ordinary sampling SE.",
    "Equal-k chance correction is mean absolute probability-point advantage over 1/k, not relative lift.",
    "A dominance front or unresolved group is not an equivalence class. Pairwise equivalence is not transitive.",
    "Planned detection power under the stated alternative does not promise decisive fronts, practical dominance or equivalence; observed power is not estimated.",
    "Preflight completion, root agreement and stable batch means do not establish comprehensive independence. Batch-only checks cannot detect dependence hidden by unchanged batch totals.",
    "Selected outcome intervals are pointwise approximate Monte Carlo precision under exchangeable independent batches, not simultaneous coverage or rare-tail guarantees. Diagnostic concerns qualify this assumption.",
    "Parameter responses/nonadditivity and permutation importance are exploratory associations, not causal or multiplicity-controlled discoveries. OOF corners may use different folds and are not a model-interaction contrast.",
    "Canonical-order examples below are excerpts, not a favorable subset or the entire population; exact full evidence remains at the linked paths. Nulls retain source statuses.",
    "Nonfinite published IEEE values use an explicit published_nonfinite token (Infinity, -Infinity or NaN), preserving their sign/type rather than substituting null or zero. Consult the source status and denominator; an infinite standardized discrepancy can reflect zero estimated reference scale, not an infinite probability effect.",
    "No new behavior, histogram-bin, rank, tail-quantile or parameter-interaction intervals are calculated.",
]


def _published_json(value: Any) -> Any:
    """Preserve source nonfinite values explicitly in strict JSON.

    Generic scalar helpers turn nonfinite numbers into nulls, which would lose
    the distinction between a published infinity and an absent estimate here.
    This encoding changes representation only, never the producer's status.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return {
            "published_nonfinite": (
                "NaN" if math.isnan(value) else "Infinity" if value > 0 else "-Infinity"
            )
        }
    if isinstance(value, dict):
        return {key: _published_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_published_json(item) for item in value]
    return value


# Aliases refer to published products, never guessed columns. The dictionary
# enumerates actual schemas at runtime; these names select products only.
ROOT_PRODUCTS: dict[str, dict[str, tuple[str, ...]]] = {
    "metrics": {
        "by_k": (
            "outcome_frequencies.parquet",
            "outcome_batch_stats.parquet",
            "outcome_batch_inventory.parquet",
            "outcome_support.parquet",
            "outcome_summaries.parquet",
            "outcome_precision.parquet",
            "seat_behavior_batch.parquet",
            "turn_opportunity_batch.parquet",
            "seat_effects.parquet",
            "seat_population_effects.parquet",
            "performance.parquet",
        ),
        "across_k": (
            "seat_effects_standardized_across_k.parquet",
            "performance_equal_k.parquet",
            "performance_bootstrap.parquet",
        ),
        "diagnostics": ("player_count_effects.parquet",),
    },
    "hgb": {"across_k": ("future_simulation_proposals.parquet",)},
    "trueskill": {"diagnostics": ("screening_diagnostics.parquet",)},
    "game_stats": {"concat_ks": ("game_length.parquet", "margin_stats.parquet")},
    "rng_diagnostics": {"diagnostics": ("preflight/publication.json",)},
}
ALIASES = {
    "outcome_frequencies": "F",
    "outcome_batch_stats": "B",
    "outcome_batch_inventory": "I",
    "outcome_support": "S",
    "outcome_summaries": "Q",
    "outcome_precision": "U",
    "seat_behavior_batch": "B7",
    "seat_behavior_summaries": "Q7",
    "turn_opportunity_batch": "O7",
    "turn_opportunity": "O",
    "turn_opportunity_equal_k": "OA",
    "parameter_responses": "R1",
    "parameter_nonadditivity": "R2",
    "parameter_response_support": "RI",
    "preflight_evidence": "PE",
    "monitor_coverage": "MC",
    "production_monitor": "MR",
    "seat_effects": "EE",
    "seat_population_effects": "EE",
    "seat_effects_standardized_across_k": "EE",
    "strategy_manifest": "POP",
    "cap_authorization": "AUTH",
    "game_length": "EG",
    "margin_stats": "EG",
    "future_simulation_proposals": "EM",
    "screening_diagnostics": "ET",
    "simulation_workload_plan": "PLAN",
    "root_order_counts": "EH",
    "combined_order_counts": "EH",
    "player_count_effects": "EK",
}


def alias(path: Path) -> str:
    stem = path.stem
    if stem == "publication" and path.parent.name == "preflight":
        return "PREF"
    if stem in ALIASES:
        return ALIASES[stem]
    if stem.startswith(("feature_importance", "heldout_")):
        return "EM"
    if stem.startswith("performance"):
        return "EP" if path.parent.parent.name == "by_k" or stem.endswith("p") else "EK"
    if stem.startswith("root_"):
        return "ER"
    if "dominance" in stem or "cycle" in stem:
        return "ED"
    if "inference" in stem or stem in {"power_plan", "order_counts"}:
        return "EH"
    if "agreement" in stem or "percentile" in stem:
        return "ET"
    if "candidate" in stem:
        return "ES"
    return "EV"


def root_stage(cfg: AppConfig, roots: tuple[int, ...], root: int, stage: str) -> Path:
    folder = cfg.root_input_stage_folder(stage)
    if folder is None:
        raise ValueError(f"analysis guide requires root stage {stage}")
    base = (
        cfg.analysis_dir
        if len(roots) == 1
        else (resolve_results_dir(base_results_dir(cfg), root) / cfg.io.analysis_subdir)
    )
    return base / folder


def guide_sources(cfg: AppConfig, roots: tuple[int, ...]) -> list[Path]:
    """Exact root-owned sources in addition to existing reporting inputs."""
    paths = [cfg.h2h_cap_authorization_path()]
    for root in roots:
        results = (
            cfg.results_root
            if len(roots) == 1
            else resolve_results_dir(base_results_dir(cfg), root)
        )
        paths.append(results / "strategy_manifest.parquet")
        paths.extend(
            results / f"{k}_players" / "simulation_workload_plan.json"
            for k in sorted(set(cfg.sim.n_players_list))
        )
        for stage, scopes in ROOT_PRODUCTS.items():
            directory = root_stage(cfg, roots, root, stage)
            for scope, names in scopes.items():
                for k in (sorted(set(cfg.sim.n_players_list)) if scope == "by_k" else [0]):
                    scope_dir = directory / scope
                    if k:
                        scope_dir /= f"{k}p"
                    paths.extend(scope_dir / name for name in names)
    if len(roots) == 2:
        paths.extend(
            [
                cfg.root_discrepancies_path(),
                cfg.root_joint_discrepancy_path(),
                cfg.root_top_n_stability_path(),
                cfg.root_shortlist_changes_path(),
                cfg.root_matched_count_convergence_path(),
                cfg.root_half_drift_path(),
            ]
        )
    return paths


def inspect_product(path: Path, budget: int) -> dict[str, Any]:
    """Authenticate and retain fixed-size examples plus closed status counts.

    F/B/B7/RI are potentially very large. Their full counts come from Parquet
    metadata, and only a bounded prefix is decoded. Certified support/precision
    summaries are scanned with bounded batches, without a roster accumulator.
    """
    metadata = validate_artifact_sidecar(path)
    result: dict[str, Any] = {
        "path": str(path),
        "alias": alias(path),
        "operation": metadata.operation,
        "conditioning": metadata.conditioning,
        "weighting": {
            "method": metadata.k_aggregation_method,
            "k_weights": metadata.k_weights,
            "quantity": metadata.weighted_quantity,
        },
        "precision": {
            "method": metadata.uncertainty_method,
            "replication_unit": metadata.replication_unit,
        },
        "claim_support": metadata.support_count_role,
        "required_player_counts": metadata.required_player_counts,
        "missing_cell_policy": metadata.missing_cell_policy,
        "sha256": metadata.artifact_sha256,
        "sidecar_sha256": sha256_file(sidecar_path(path)),
        "code_revision": metadata.code_revision,
        "config_hash": metadata.config_hash,
        "method_contract": metadata.method_contract,
    }
    if metadata.artifact_contract_version == 3:
        authenticated = load_authenticated_sidecar(path)
        result["code_identity"] = asdict(authenticated.stage_identity.code)
        result["config_identity"] = asdict(authenticated.stage_identity.stage_config)
        result["versions"] = asdict(authenticated.versions)
    if path.suffix == ".json":
        payload = read_json_file_with_retry(path)
        if not isinstance(payload, dict):
            raise ValueError(f"guide requires JSON object: {path}")
        result.update(
            columns=list(payload),
            published_values={k: v for k, v in payload.items() if k != "sources"},
            rows=None,
        )
        result["values_omitted"] = ["sources"] if "sources" in payload else []
        return cast(dict[str, Any], _published_json(result))
    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    result["columns"] = schema.names
    result["column_types"] = {f.name: str(f.type) for f in schema}
    result["rows"] = parquet.metadata.num_rows
    if result["alias"] in {"S", "Q", "U", "POP"} and result["rows"] == 0:
        raise ValueError(f"missing required support rows in {path}")
    result["examples"] = []
    result["status_counts"] = {}
    scan_status = [n for n in ("status", "support_status", "result") if n in schema.names]
    large = result["alias"] in {"F", "B", "B7", "RI"}
    prefix_only = result["alias"] == "Q7"
    counts: dict[str, Counter[str]] = {n: Counter() for n in scan_status}
    precision_range: dict[str, list[float]] = {}
    variants: set[tuple[Any, ...]] = set()
    variant_columns = [
        n
        for n in (
            "quantity",
            "statistic",
            "conditioning",
            "summary_id",
            "metric",
            "parameter",
            "parameter_a",
            "parameter_b",
            "status",
        )
        if n in schema.names
    ]
    for _, _, table in iter_parquet_tables_by_bytes(
        path,
        columns=schema.names,
        max_batch_bytes=max(1, budget // 4),
        max_batch_rows=(
            min(256, max(1, budget // 32768)) if prefix_only else max(1, budget // 32768)
        ),
    ):
        needed = SAMPLE_ROWS - len(result["examples"])
        if needed > 0:
            result["examples"].extend(_published_json(table.slice(0, needed).to_pylist()))
        if result["alias"] in {"Q", "U", "R1", "R2", "Q7"}:
            for example in _published_json(table.to_pylist()):
                variant = tuple(example[n] for n in variant_columns)
                if variant not in variants and len(variants) < 128:
                    variants.add(variant)
                    if example not in result["examples"]:
                        result["examples"].append(example)
        if large:
            break
        for name in scan_status:
            for record in cast(Any, pc.value_counts(table[name])).to_pylist():
                counts[name][str(record["values"])] += record["counts"]
                if len(counts[name]) > 64:
                    raise ValueError(f"unbounded status vocabulary in {path}")
        if result["alias"] == "U":
            for name in ("interval_width", "mcse", "denominator", "contributing_batches"):
                bounds = cast(Any, pc.min_max(table[name])).as_py()
                if bounds["min"] is not None:
                    low, high = bounds["min"], bounds["max"]
                    previous = precision_range.get(name, [low, high])
                    precision_range[name] = [min(previous[0], low), max(previous[1], high)]
        if prefix_only:
            break
    result["status_counts"] = {n: dict(sorted(c.items())) for n, c in counts.items()}
    result["status_count_scope"] = (
        "not_scanned_large_inventory"
        if large
        else "bounded_prefix" if prefix_only else "entire_artifact"
    )
    if precision_range:
        result["published_precision_ranges"] = precision_range
    result["example_selection"] = (
        f"first {SAMPLE_ROWS} rows plus first representative of at most 128 metric/conditioning/status variants in published order; full artifact has {result['rows']} rows"
    )
    result["visual_fields"] = [
        n
        for n in schema.names
        if n
        in {
            "strategy",
            "root_seed",
            "k",
            "seat",
            "distribution",
            "value",
            "value_scale",
            "count",
            "estimate",
            "denominator",
            "ci_low",
            "ci_high",
            "interval_width",
            "mcse",
            "status",
            "chance_delta",
            "win_rate",
            "equal_k_score",
            "across_k_score",
            "practical_front",
            "parameter",
            "level",
            "observed_mean",
            "oof_predicted_mean",
            "matched_backgrounds",
            "observed_nonadditivity",
            "support_id",
            "fold",
            "mae",
            "r2",
            "feature",
            "association_importance_mean",
            "mean_turn_range",
            "turn_share",
            "metric",
            "conditioning",
        }
    ]
    return cast(dict[str, Any], _published_json(result))


# Method descriptions are definitions, not assertions about scientific results.
METHODS = {
    "performance": {
        "computation": "W/A and p_k-1/k; descriptive screening order; complete per-k profile, Pareto/maximin from published performance.",
        "estimand": "Win probability against the configured finite tournament mixture and absolute chance-relative advantage.",
        "conditioning": "All attempted player-game exposures; safety attempts count as losses. Completed-only W/C is a separate diagnostic.",
        "denominator": "A attempted exposures; C only for labeled completed diagnostics.",
        "weighting": "Within k use exposure counts; across k use published declared weights (equal-k default); roots retain source weights. Unequal exposure mixing is secondary.",
        "precision": "Published attempted Wilson/batch MCSE and shared-vector bootstrap intervals; rankings/Pareto/maximin remain descriptive.",
    },
    "outcome": {
        "computation": "Exact empirical frequencies; inverse empirical CDF Q(q)=min{x:F(x)>=q} at .05/.25/.50/.75/.95. Recorded winner/ranks remain official; descriptive first ties add no wins.",
        "estimand": "Completed final-score/rank distribution, completed deficit D/C, losing severity D/L, winning margin V/W; stopped scores are separate.",
        "conditioning": "Completed exposures for final scores/ranks; official completed losses/wins for conditional margins; safety_limit for stopped scores.",
        "denominator": "C for completed scores/ranks/deficit, L for losing margin, W for winning margin, S for stopped scores. Display W/C and L/C alongside severity.",
        "weighting": "Each actual-seat exposure counts once; repeated strategy IDs are separate exposures. Seat=0 sums actual seats before division. No cross-k/root outcome interval.",
        "precision": "U retains five selected achieved-precision summaries: completed win/loss, D/C, D/L, V/W. Pointwise deterministic-batch percentile intervals and ddof=1 MCSE; shared root/k draws. Source statuses withhold intervals for fewer than 30 total/contributing batches, fewer than 2000 replicates, or sparse/degenerate/undefined resamples. Frequencies, ranks, quantiles and tails are descriptive only.",
    },
    "h2h": {
        "computation": "Order-stratified score inference for d_ab; balanced A win probability .5+d_ab. Dominance graphs/fronts preserve cycles and incomparability; Holm superiority, practical dominance and strict pairwise equivalence are independent decisions.",
        "estimand": "Direct balanced two-player comparison within the frozen selected family.",
        "conditioning": "Completed H2H games for inference; attempted support/completion viability separately retained; no-test pairs remain in the multiplicity family.",
        "denominator": "Completed games by pair/root/order; attempted games for completion/safety rates.",
        "weighting": "Equal roots and both seat orders as specified by the immutable power plan; retain q_ab/q_ba and order support.",
        "precision": "Ordinary and Bonferroni simultaneous effect bounds/widths; configured Holm family alpha for superiority; practical threshold and equivalence tolerance separately. Planned target power applies to the plan's alternative, not observed effects or decisive fronts.",
    },
    "behavior": {
        "computation": "Published terminal behavior, actual-turn efficiency, round proxy gap/mismatch, seat effects, and once-per-game turn opportunity.",
        "estimand": "Descriptive opportunity-normalized behavior and seat-conditioned outcomes; turn opportunity is not a seat-adjusted win estimator.",
        "conditioning": "Completed primary behavior/opportunity; safety_limit and all_attempted views separately labeled. Completed turns/rounds must be positive; safety zero-turn with zero score contributes zero to the game-ratio mean.",
        "denominator": "Points/turn: sum(score)/sum(turns); mean game rate/proxy/gap/mismatch/highest turn: exposures. Count rates: actual turns or exposures. Heuristic dice/use: recorded uses; null without uses. Turn range counts each completed game once; shares divide by all completed seat turns.",
        "weighting": "Ratio-of-sums weights turns; mean-of-game-ratios weights exposures. Seat=0 sums actual seats. OA is equal-k mean of completed mean turn range separately by root; every configured k required, no renormalization.",
        "precision": "Retain published seat win precision separately; behavior/score/rank/margin seat differences and opportunity are descriptive, with no new contrast family or behavior CI.",
    },
    "roots": {
        "computation": "Published root discrepancy/rank/top-N/shortlist/convergence/half-drift summaries and joint batch references, alongside required diagnostic coverage.",
        "estimand": "Fixed-design descriptive reproducibility for two declared roots; no hypothetical seed-population inference.",
        "conditioning": "Configured rules/grid/k support; H2H root diagnostics condition on completed frozen-family games.",
        "denominator": "Published root/k attempted exposures and certified deterministic batches; diagnostic support is check-specific.",
        "weighting": "Identical common k support and published root/k weights; no favorable filtering.",
        "precision": "Source Monte Carlo precision and joint discrepancy reference frequencies are descriptive, not heterogeneity or multiple-testing inference. Root agreement is not comprehensive independence evidence.",
    },
    "parameters": {
        "computation": "Ten bounded active-parameter response tables; three prespecified adjacent-level quartets. Observed p11-p10-p01+p00 is separate from OOF corner means; RI enumerates exact matched strategy/fold identities.",
        "estimand": "Finite-grid observed all-attempt win-probability associations and held-out HGB predictive associations; proposed strategies are untested candidates.",
        "conditioning": "Valid canonical settings at fixed root/k; one-way backgrounds support all selected levels; quartets require all four corners at identical other attributes. Inactive sentinels/equivalent settings do not add support.",
        "denominator": "Matched backgrounds for response/corner means; attempted exposure denominator for source probabilities; held-out strategies for fold MAE/R2 and permutation associations.",
        "weighting": "Equal matched-background weights; selected levels follow the frozen <=25-level rule. Folds hold out whole strategies. Different rectangles may have different exact support.",
        "precision": "Held-out fold metrics and permutation importance variation are predictive diagnostics. No inferential interaction test, causal effect, cross-fold model interaction, or multiplicity-controlled discovery.",
    },
    "diagnostics": {
        "computation": "Authenticated fault-to-check correctness/monitor coverage, minimum support, seeded clean/fault capability and production result records; explicit optional exhaustive-lag scope.",
        "estimand": "Declared diagnostic capability under bounded synthetic fault strengths plus actual production-monitor observations; correctness preflight observations are separate.",
        "conditioning": "Frozen validation recipe and current code/config/environment applicability; production monitors select <=16 canonical strategies, all-seat vectors per required root/k.",
        "denominator": "64 seeded fixture families per scenario/B, not independent check opportunities. Strong capability fixtures use AR(1) coefficients +.85/-.85 or a 4-latent-SD half step; weak fixtures use +.2 or a .5-latent-SD step, with measured misses retained. Statistical monitor v1 assesses B=30 or 100, positive denominators, nonconstant series; at least 30 batches alone is insufficient.",
        "weighting": "Check-specific support; retain every batch in order and every undefined status. No dropping unsupported batches or turning missing artifacts into zeros.",
        "precision": "Lag abs(rho)>max(.5,4/sqrt(B-1)) and half drift >1.5 are descriptive effect thresholds. Measured clean alerts do not calibrate per-check or dataset-wide false-positive control; no new statistical stopping rule.",
    },
}


def question_spec(cell: str) -> tuple[str, set[str]]:
    row = int(cell[1:])
    if row in {7}:
        return "performance", {"ET", "EP", "ES"}
    if 20 <= row <= 24:
        return "outcome", {"F", "I", "S", "Q", "U"}
    if row in {32} or 40 <= row <= 48 or row in {55, 56, 78}:
        return "h2h", {"ED", "EH", "ET", "AUTH"}
    if 52 <= row <= 71:
        return "behavior", {"EE", "Q7", "B7", "O", "O7", "OA", "Q", "U", "EG"}
    if row in {80, 100}:
        return "diagnostics", {"PE", "MC", "MR", "PREF"}
    if 75 <= row <= 79:
        return "roots", {"ER", "ET"}
    if 87 <= row <= 91:
        return "parameters", {"EM", "R1", "R2", "RI"}
    if row >= 97:
        return "diagnostics", {"POP", "PLAN", "I", "S", "U", "PE", "MC", "MR", "PREF", "AUTH", "EH"}
    return "performance", {"EP", "EK", "ES", "ER"}


QUESTION_ALIASES = {
    3: {"EP"},
    4: {"EP", "Q", "U"},
    5: {"EP", "EK", "ES"},
    6: {"EK", "ER", "EE", "Q"},
    7: {"ET", "EP", "ES"},
    11: {"EP", "EK"},
    12: {"EP", "EK"},
    13: {"EK", "ER"},
    14: {"EK"},
    15: {"EK"},
    16: {"EP", "EK"},
    20: {"F", "I", "S", "Q"},
    21: {"F", "S", "Q"},
    22: {"F", "S", "Q", "U"},
    23: {"F", "Q", "U"},
    24: {"F", "B", "I", "S", "Q"},
    28: {"EP", "POP"},
    30: {"EP", "EK"},
    52: {"EE"},
    53: {"EE"},
    54: {"EE", "Q", "U"},
    55: {"EH"},
    56: {"EH"},
    57: {"O", "OA", "O7"},
    61: {"Q7", "B7"},
    62: {"Q7", "B7"},
    63: {"Q7"},
    67: {"Q7"},
    68: {"Q7", "EG"},
    69: {"Q7", "EM"},
    70: {"Q7"},
    71: {"Q7", "B7"},
    75: {"ER"},
    76: {"ER", "ES"},
    77: {"ER", "EK"},
    87: {"EM", "R1"},
    88: {"EM", "R1"},
    89: {"R1", "R2", "RI"},
    90: {"EM"},
    91: {"EM"},
    98: {"POP", "PLAN", "AUTH"},
    99: {"I", "S", "RI"},
    101: {"U", "EP", "EH"},
    102: {"EH", "ED", "Q", "S", "RI"},
}
QUESTION_METRICS = {
    20: {"completed_score", "stopped_score"},
    21: {"competition_rank", "midrank"},
    22: {"losing_margin", "mean_losing_margin", "completed_loss_rate", "mean_deficit_completed"},
    61: {"points_per_turn"},
    62: {"mean_game_points_per_turn", "zero_turn_rate"},
    63: {
        "mean_game_points_per_turn",
        "mean_game_points_per_round",
        "round_proxy_gap",
        "turn_round_mismatch_rate",
    },
    68: {"mean_rounds", "mean_turns"},
}


def question_examples(product: dict[str, Any], row: int) -> list[dict[str, Any]]:
    examples = product.get("examples", [])
    wanted = QUESTION_METRICS.get(row)
    if wanted is not None:
        return [
            example
            for example in examples
            if any(
                example.get(n) in wanted
                for n in ("quantity", "metric", "summary_id", "distribution")
            )
            or not any(n in example for n in ("quantity", "metric", "summary_id", "distribution"))
        ]
    return examples


def build_guide(
    cfg: AppConfig,
    report: dict[str, Any],
    sources: list[Path],
    *,
    oracle_game_profile: Any | None = None,
) -> dict[str, Any]:
    """Build the only result model; Markdown never accesses producers or config."""
    dictionary = [
        inspect_product(p, cfg.resources.stage_batch_bytes["analysis"])
        for p in sorted(set(sources), key=str)
    ]
    if cfg._game_profile_sha256 is not None and (
        oracle_game_profile is None or oracle_game_profile.sha256 != cfg._game_profile_sha256
    ):
        raise ValueError("analysis guide requires the authenticated run's game profile")
    h2h = dict(report["h2h"])
    power_plan = next(
        p["published_values"] for p in dictionary if p["operation"] == "score_test_power_plan"
    )
    h2h["assumed_alternative"] = {
        k: v
        for k, v in power_plan.items()
        if k
        in {
            "target_effect",
            "seat1_advantage_scenarios",
            "seat_scenarios",
            "sensitivity_deltas",
            "sensitivity",
            "scenario_results",
            "power_validation",
            "target_power",
            "family_alpha",
        }
    }
    h2h["supporting_pairwise_detail"] = [dict(row) for row in h2h.pop("pair_intervals")]
    for row in h2h["supporting_pairwise_detail"]:
        for prefix in ("ordinary", "simultaneous"):
            low, high = row[f"{prefix}_d_low"], row[f"{prefix}_d_high"]
            row[f"{prefix}_interval_width"] = None if low is None or high is None else high - low
    groups: list[dict[str, Any]] = [
        {"heading_cell": f"A{start}", "title": title, "questions": []} for start, title in GROUPS
    ]
    notes = []
    for cell, question, _historical_coverage, boundary in QUESTIONS:
        row = int(cell[1:])
        group_index = max(i for i, (start, _) in enumerate(GROUPS) if start <= row)
        method, aliases = question_spec(cell)
        aliases = QUESTION_ALIASES.get(row, aliases)
        status = (
            "deferred"
            if row in {29, 31, 81}
            else "excluded" if row == 97 else "partial" if row in {6, 30, 70} else "available"
        )
        if len(report["roots"]) == 1 and 75 <= row <= 79:
            status = "unestimable"
        evidence = (
            []
            if status in {"deferred", "excluded"}
            else [p for p in dictionary if p["alias"] in aliases]
        )
        if status not in {"deferred", "excluded"} and not evidence:
            raise ValueError(f"missing required guide evidence for {cell}")
        workbook_cells = [cell]
        note_cells = {
            3: ["B3"],
            4: ["B4", "C4"],
            5: ["B5"],
            6: ["B6"],
            7: ["B7"],
            11: ["B11"],
            12: ["B12"],
            13: ["B13"],
            16: ["B16"],
        }.get(row, [])
        workbook_cells.extend(note_cells)
        if note_cells:
            notes.append(
                {"workbook_cells": note_cells, "question_cell": cell, "resolution": boundary}
            )
        entry: dict[str, Any] = {
            "workbook_cells": workbook_cells,
            "question": question,
            "status": status,
            "status_axis": "Question coverage; individual quantities retain their published support and precision statuses. A single root cannot establish two-root reproducibility.",
            **METHODS[method],
            "claim_limits": [boundary, METHODS[method]["precision"]],
            "evidence": [{"path": p["path"], "columns": p["columns"]} for p in evidence],
            "results": [
                {
                    "path": p["path"],
                    "rows": p["rows"],
                    "status_counts": p.get("status_counts", {}),
                    "status_count_scope": p.get("status_count_scope", "not_applicable_json"),
                    "published_precision_ranges": p.get("published_precision_ranges", {}),
                    "examples": question_examples(p, row),
                    "values_reference": (
                        "artifact_dictionary.published_values" if "published_values" in p else None
                    ),
                }
                for p in evidence
            ],
            "support": "Published row counts and exact source denominators/statuses below; example rows are bounded excerpts. Full results and identities are in the artifact dictionary.",
        }
        if status in {"deferred", "excluded"}:
            entry.update(
                computation="No first-release estimator for this question.",
                estimand="Not estimated.",
                conditioning="Not applicable.",
                denominator="Not applicable.",
                weighting="Not applicable.",
                precision="Not applicable.",
            )
        if row == 57:
            entry["headline"] = report["behavior_opportunity_parameters"]["overall_opportunity"]
        elif row in {6, 12, 13, 15}:
            entry["headline"] = report["robustness"]
        elif method == "behavior":
            entry["headline"] = [
                {
                    "root_seed": c["root_seed"],
                    "k": c["k"],
                    "behavior": [
                        r
                        for r in c["behavior"]
                        if row not in QUESTION_METRICS or r["metric"] in QUESTION_METRICS[row]
                    ],
                    "opportunity": c["opportunity"],
                }
                for c in report["behavior_opportunity_parameters"]["root_k_cells"]
            ]
        elif method == "parameters":
            entry["headline"] = [
                {
                    "root_seed": c["root_seed"],
                    "k": c["k"],
                    "exact_product_support": c["parameters"],
                    "heldout_folds": c["heldout_folds"],
                    "permutation_association_importance": c["permutation_association_importance"],
                }
                for c in report["behavior_opportunity_parameters"]["root_k_cells"]
            ]
        groups[group_index]["questions"].append(entry)
    identity = {
        "roots": report["roots"],
        "player_counts": report["support"]["player_counts"],
        "profile": report["profile"],
        "execution_scope": report["execution_scope"],
        "configured_strategy_population": asdict(cfg.sim),
        "rules_and_workload_evidence": [
            p["path"] for p in dictionary if p["alias"] in {"POP", "PLAN", "I"}
        ],
        "population_evidence": [p for p in dictionary if p["alias"] == "POP"],
        "code_config_identities": [
            {
                "path": p["path"],
                "code_identity": p.get("code_identity", p["code_revision"]),
                "config_identity": p.get("config_identity", p["config_hash"]),
            }
            for p in dictionary
        ],
        "authorizations": [p for p in dictionary if p["alias"] == "AUTH"],
        "design_conditioning": report["conditioning"],
        "weights": report["support"],
        "rules": {
            "target_score": (
                10000 if oracle_game_profile is None else oracle_game_profile.default_target_score
            ),
            "max_rounds": (
                200 if oracle_game_profile is None else oracle_game_profile.default_max_rounds
            ),
            "scoring_and_ties": "Production engine/scoring rules bound to the code identity; official winner tie policy and final round, no winner/ranks at safety termination.",
            "game_profile": (
                None if oracle_game_profile is None else oracle_game_profile.canonical_payload()
            ),
            "game_profile_sha256": cfg._game_profile_sha256,
        },
    }
    # Paths/tuples from public config are normalized once into the JSON model.
    identity = json.loads(json.dumps(identity, default=str))
    guide = {
        "schema_version": SCHEMA_VERSION,
        "contract_id": CONTRACT_ID,
        "run_identity": identity,
        "question_groups": groups,
        "methodological_notes": notes,
        "artifact_dictionary": dictionary,
        "coverage": {
            "question_count": len(QUESTIONS),
            "group_count": len(groups),
            "question_status_counts": dict(
                Counter(q["status"] for g in groups for q in g["questions"])
            ),
            "workbook": {
                "name": "Farkle Questions.xlsx",
                "sheet": "Sheet1",
                "sha256": "5bdecfbd895ce62e3047abcc3bcee52f9b5f3028c7e7a6c017461b3286d3502c",
                "label_cells": [f"A{s}" for s, _ in GROUPS] + ["C1"],
                "framing_cells": FRAMING,
            },
            "status_dictionary": {
                "available": "Required authenticated products present; individual estimates/precision retain their own source statuses.",
                "partial": "Covered dimensions are shown; excluded component remains unanswered.",
                "verified_zero": "Only producer-certified complete support with zero exposures; not missing input.",
                "missing_required_data": "Blocks successful guide/stage publication; never converted to zero or a successful empty guide.",
                "unestimable": "Unsupported matched backgrounds or zero conditional denominator; estimates null, retain exact reason.",
                "insufficient_evidence": "Check or precision support insufficient; no reassurance or invented zero.",
                "deferred": "No first-release estimator.",
                "excluded": "Intentionally outside first release.",
                "omitted_by_design": "Optional diagnostic deliberately omitted; it did not pass or run.",
            },
            "diagnostics": [p for p in dictionary if p["alias"] in {"PE", "MC", "MR", "PREF"}],
            "behavior_opportunity_parameters": report["behavior_opportunity_parameters"],
            "h2h": h2h,
        },
        "limitations": LIMITATIONS,
    }
    return cast(
        dict[str, Any],
        json.loads(
            json.dumps(_published_json(guide), ensure_ascii=False, allow_nan=False, sort_keys=True)
        ),
    )


def _render_value(value: Any) -> list[str]:
    """Render arbitrary source values without computing or interpreting them."""
    return [
        "```json",
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, sort_keys=True),
        "```",
        "",
    ]


def _table_rows(rows: list[dict[str, Any]]) -> list[str]:
    """Render published rows with deterministic column order."""
    if not rows:
        return ["No rows in this excerpt; consult the source support/status record.", ""]
    columns = sorted({key for row in rows for key in row})

    def cell(value: Any) -> str:
        return (
            json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
            .replace("|", "\\|")
            .replace("\n", " ")
        )

    return [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
        *["| " + " | ".join(cell(row.get(key)) for key in columns) + " |" for row in rows],
        "",
    ]


def _result_lines(results: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for result in results:
        lines += [
            f"[{Path(result['path']).name}](<{Path(result['path']).as_posix()}>): {result['rows']} published rows; statuses ({result['status_count_scope']}) `{json.dumps(result['status_counts'], sort_keys=True)}`.",
            "",
        ]
        if result["published_precision_ranges"]:
            lines += [
                f"Published precision/support ranges: `{json.dumps(result['published_precision_ranges'], sort_keys=True)}`.",
                "",
            ]
        if result["examples"]:
            lines += _table_rows(result["examples"])
        if result["values_reference"]:
            lines += [
                "Actual JSON values and method metadata are in the artifact dictionary below.",
                "",
            ]
    return lines


def _published_lines(payload: dict[str, Any]) -> list[str]:
    """Keep diagnostic envelopes readable while retaining all check records."""
    if "checks" not in payload:
        return _render_value(payload)
    lines = _render_value({key: value for key, value in payload.items() if key != "checks"})
    fields = {
        "check_id",
        "kind",
        "execution_status",
        "coverage",
        "minimum_support",
        "support",
        "statistic",
        "effect_threshold",
        "result",
        "reason",
        "fault_recipe_id",
        "clean_fixture_count",
        "fault_fixture_count",
        "clean_alert_count",
        "detected_fault_count",
    }
    lines += [
        "All published check records (shared applicability/blind spots above; complete records in JSON):",
        "",
        *_table_rows(
            [
                {key: value for key, value in check.items() if key in fields}
                for check in payload["checks"]
            ]
        ),
    ]
    return lines


def render_guide_markdown(guide: dict[str, Any]) -> str:
    """Pure rendering of the JSON model, including every result/support field."""
    lines = [
        "# Analysis guide",
        "",
        f"Contract: `{guide['contract_id']}`; schema `{guide['schema_version']}`.",
        "",
        "## Run identity",
        "",
        *_render_value(
            {
                key: value
                for key, value in guide["run_identity"].items()
                if key not in {"population_evidence", "code_config_identities", "authorizations"}
            }
        ),
    ]
    for group in guide["question_groups"]:
        lines += [f"## {group['heading_cell']}: {group['title']}", ""]
        if group["heading_cell"] == "A36":
            h2h = guide["coverage"]["h2h"]
            for name, value in (
                ("Dominance fronts", h2h["fronts"]),
                ("Cycles", h2h["cycles"]),
                (
                    "Unresolved and explicit pairwise equivalence",
                    {
                        k: h2h[k]
                        for k in (
                            "unresolved_pair_count",
                            "unresolved_nonviable_pair_count",
                            "equivalent_pair_count",
                            "delta_equivalence",
                        )
                    },
                ),
                (
                    "Planned detection power and assumed alternative",
                    {
                        k: h2h[k]
                        for k in (
                            "planned_detection_power",
                            "planned_worst_scenario_power",
                            "assumed_alternative",
                            "method_notes",
                        )
                    },
                ),
                (
                    "Attained support",
                    {
                        k: h2h[k]
                        for k in (
                            "games_attempted",
                            "games_completed",
                            "games_safety_limit",
                            "by_pair_root_order",
                            "candidate_viability",
                        )
                    },
                ),
            ):
                lines += [f"### {name}", "", *_render_value(value)]
            lines += [
                "A front or unresolved group is not an equivalence class. Pairwise intervals and independent statistical/practical/equivalence flags are supporting detail.",
                "",
                "<details><summary>Pairwise bounds, widths and decisions</summary>",
                "",
                *_render_value(h2h["supporting_pairwise_detail"]),
                "</details>",
                "",
            ]
        for q in group["questions"]:
            lines += [
                f"### {', '.join(q['workbook_cells'])}: {q['question']}",
                "",
                f"Coverage: **{q['status']}**.",
                "",
            ]
            for key in (
                "computation",
                "estimand",
                "conditioning",
                "denominator",
                "weighting",
                "precision",
                "support",
            ):
                lines += [f"{key.replace('_', ' ').capitalize()}: {q[key]}", ""]
            if "headline" in q:
                lines += ["Headline evidence:", "", *_render_value(q["headline"])]
            lines += [
                "Results and supporting counts (canonical excerpts):",
                "",
                *_result_lines(q["results"]),
            ]
            for evidence in q["evidence"]:
                lines += [
                    f"- [Evidence](<{Path(evidence['path']).as_posix()}>): `{', '.join(evidence['columns'])}`."
                ]
            lines += ["", "Claim limits:", "", *[f"- {v}" for v in q["claim_limits"]], ""]
    lines += [
        "## Methodological-note resolutions",
        "",
        *_table_rows(guide["methodological_notes"]),
        "## Artifact and column dictionary",
        "",
        "Use the listed visual fields with their identity, conditioning, denominator and status fields. These are examples for manual visuals.",
        "",
    ]
    for product in guide["artifact_dictionary"]:
        lines += [
            f"### {product['alias']}: [{Path(product['path']).name}](<{Path(product['path']).as_posix()}>)",
            "",
            f"Published columns: `{', '.join(product['columns'])}`.",
            "",
            f"Operation: `{product['operation']}`. Conditioning: {product['conditioning']}.",
            "",
            f"Weighting: `{json.dumps(product['weighting'], sort_keys=True)}`. Precision: `{json.dumps(product['precision'], sort_keys=True)}`.",
            "",
            f"Rows: `{product['rows']}`. Method/config/code and exact byte identities are retained in the JSON dictionary.",
            "",
            f"Artifact SHA256: `{product['sha256']}`. Config identity: `{product['config_hash']}`. Code revision: `{product['code_revision']}`.",
            "",
            f"Manual visual fields: `{', '.join(product.get('visual_fields', []))}`.",
            "",
            *(
                _published_lines(product["published_values"])
                if "published_values" in product
                else []
            ),
        ]
    lines += [
        "## Coverage and diagnostic evidence",
        "",
        *_render_value(
            {
                key: value
                for key, value in guide["coverage"].items()
                if key not in {"h2h", "behavior_opportunity_parameters", "diagnostics"}
            }
        ),
        "## Limitations and exclusions",
        "",
        *[f"- {v}" for v in guide["limitations"]],
        "",
    ]
    return "\n".join(lines)


def guide_product_reusable(
    cfg: AppConfig, path: Path, payload: Any, expected_metadata: ArtifactSidecar
) -> bool:
    """Retain valid guide children during a stale-parent or interrupted replay.

    The freshly extracted JSON binds every input byte/sidecar and identity.
    Equal payload plus current authenticated output code/method prevents reuse
    under changed provenance; a missing sidecar is never promoted.
    """
    try:
        metadata = validate_artifact_sidecar(
            path, expected={"operation": "render_analysis_guide_v1"}
        )
        if metadata.artifact_contract_version != 3:
            return False
        authenticated = load_authenticated_sidecar(path)
        captured = capture_v3_inputs(expected_metadata)
        if (
            authenticated.source_artifacts != captured.sources
            or authenticated.manifest_roots != captured.manifests
        ):
            return False
        if authenticated.stage_identity.stage_config != stage_config_identity(
            cfg,
            stage_key="reporting",
            field_paths=authenticated.stage_identity.stage_config.field_paths,
        ):
            return False
        current_code = cfg._code_identity or resolve_code_identity(
            Path(__file__).resolve().parents[3],
            policy=CodeIdentityPolicy.DEVELOPMENT_DIRTY,
        )
        if (
            authenticated.stage_identity.code != current_code
            or authenticated.versions.method_versions.get("analysis_guide_method_version") != 1
            or authenticated.versions.method_versions.get("structure_report_contract_version") != 9
        ):
            return False
        actual = (
            read_json_file_with_retry(path)
            if path.suffix == ".json"
            else path.read_text(encoding="utf-8")
        )
        return bool(actual == payload)
    except (OSError, ValueError, RuntimeError):
        return False


def validate_guide(guide: dict[str, Any]) -> None:
    """Check the frozen coverage/reference contract independently of rendering."""
    if set(guide) != {
        "schema_version",
        "contract_id",
        "run_identity",
        "question_groups",
        "methodological_notes",
        "artifact_dictionary",
        "coverage",
        "limitations",
    }:
        raise ValueError("analysis guide envelope differs from Task 0")
    if guide["schema_version"] != SCHEMA_VERSION or guide["contract_id"] != CONTRACT_ID:
        raise ValueError("analysis guide identity differs from Task 0")
    groups = guide["question_groups"]
    if [(g["heading_cell"], g["title"]) for g in groups] != [
        (f"A{s}", title) for s, title in GROUPS
    ]:
        raise ValueError("analysis guide requires all ten question groups")
    cells = [q["workbook_cells"][0] for g in groups for q in g["questions"]]
    if cells != [row[0] for row in QUESTIONS]:
        raise ValueError("analysis guide lost or duplicated a workbook question")
    dictionary = {p["path"]: p for p in guide["artifact_dictionary"]}
    if len(dictionary) != len(guide["artifact_dictionary"]):
        raise ValueError("guide has duplicate artifact references")
    notes = {cell for note in guide["methodological_notes"] for cell in note["workbook_cells"]}
    if notes != {"B3", "B4", "C4", "B5", "B6", "B7", "B11", "B12", "B13", "B16"}:
        raise ValueError("guide lost workbook method-note resolutions")
    for group in groups:
        for q in group["questions"]:
            if q["status"] not in {"available", "partial", "deferred", "excluded", "unestimable"}:
                raise ValueError("missing required data cannot be successful guide coverage")
            if q["status"] in {"available", "partial", "unestimable"} and not q["evidence"]:
                raise ValueError("guide requires actual evidence")
            for evidence in q["evidence"]:
                if (
                    evidence["path"] not in dictionary
                    or evidence["columns"] != dictionary[evidence["path"]]["columns"]
                ):
                    raise ValueError("guide references unpublished artifact/columns")
            for result in q["results"]:
                product = dictionary[result["path"]]
                if result["rows"] != product["rows"] or any(
                    row not in product.get("examples", []) for row in result["examples"]
                ):
                    raise ValueError("guide result differs from published evidence")
    json.dumps(guide, allow_nan=False)
