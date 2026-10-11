"""Independent guide/source reconciliation in the real two-root workflow."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pyarrow.parquet as pq

from farkle.analysis.analysis_guide import render_guide_markdown, validate_guide
from farkle.orchestration.run_contexts import RootPairRunContext, SeedRunContext
from farkle.utils.artifact_contract import sha256_file
from farkle.utils.authenticated_contract import load_authenticated_sidecar


def assert_task8_guide(
    contexts: tuple[SeedRunContext, SeedRunContext], pair: RootPairRunContext
) -> None:
    cfg = pair.config
    guide = json.loads(cfg.analysis_guide_json_path().read_text(encoding="utf-8"))
    validate_guide(guide)
    assert cfg.analysis_guide_markdown_path().read_text(encoding="utf-8") == render_guide_markdown(
        guide
    )
    dictionary = {p["path"]: p for p in guide["artifact_dictionary"]}
    assert guide["run_identity"]["roots"] == [11, 22]
    assert guide["run_identity"]["rules"]["target_score"] == 100
    assert guide["run_identity"]["rules"]["game_profile_sha256"] == cfg._game_profile_sha256
    for context in contexts:
        for k in context.config.sim.n_players_list:
            for getter in (
                context.config.outcome_support_path,
                context.config.outcome_summaries_path,
                context.config.outcome_precision_path,
                context.config.parameter_responses_path,
                context.config.parameter_nonadditivity_path,
            ):
                path = getter(k)
                product = dictionary[str(path)]
                table = pq.read_table(path)
                assert product["columns"] == table.schema.names
                assert product["rows"] == table.num_rows
                assert product["sha256"] == sha256_file(path)
                rows = table.to_pylist()
                assert all(row in rows for row in product["examples"])
                for column, counts in product["status_counts"].items():
                    assert counts == {
                        value: sum(row[column] == value for row in rows)
                        for value in sorted({row[column] for row in rows})
                    }
            coverage = dictionary[str(context.config.rng_monitor_coverage_path())][
                "published_values"
            ]
            actual = json.loads(context.config.rng_monitor_coverage_path().read_text())
            assert coverage["capability_assessment"] == actual["capability_assessment"]
            assert coverage["exhaustive_status"] == "omitted_by_design"
            assert coverage["checks"] == actual["checks"]
    h2h = guide["coverage"]["h2h"]
    actual_pairs = {
        r["pair_id"]: r for r in pq.read_table(cfg.h2h_pairwise_inference_path()).to_pylist()
    }
    for row in h2h["supporting_pairwise_detail"]:
        actual_pair = actual_pairs[row["pair_id"]]
        for name in (
            "games_attempted",
            "games_completed",
            "games_safety_limit",
            "equivalent",
            "unresolved",
            "simultaneous_d_low",
            "simultaneous_d_high",
        ):
            assert row[name] == actual_pair[name]
    authorization = json.loads(cfg.h2h_cap_authorization_path().read_text())
    assert guide["run_identity"]["authorizations"][0]["published_values"] == authorization
    md = load_authenticated_sidecar(cfg.analysis_guide_markdown_path())
    source = md.source_artifacts[0].artifact
    assert source.content_sha256 == sha256_file(cfg.analysis_guide_json_path())
    assert source == load_authenticated_sidecar(cfg.analysis_guide_json_path()).artifact
    # All paths in the consumer dictionary are explicit authenticated current-run inputs.
    assert all(Path(p).is_file() for p in dictionary)
    # A real zero-reference-scale discrepancy is infinite. Retain its signed
    # source value through strict JSON rather than imputing null or zero.
    discrepancy_path = cfg.root_discrepancies_path()
    actual_discrepancies = pq.read_table(discrepancy_path).to_pylist()
    represented = dictionary[str(discrepancy_path)]["examples"]
    nonfinite_found = False
    for example in represented:
        matching = next(
            row
            for row in actual_discrepancies
            if all(
                example[name] == value
                for name, value in row.items()
                if not isinstance(value, float) or math.isfinite(value)
            )
        )
        for name, value in matching.items():
            if isinstance(value, float) and not math.isfinite(value):
                nonfinite_found = True
                expected = "NaN" if math.isnan(value) else "Infinity" if value > 0 else "-Infinity"
                assert example[name] == {"published_nonfinite": expected}
    assert nonfinite_found, "bounded raw oracle should exercise the zero-scale discrepancy"
