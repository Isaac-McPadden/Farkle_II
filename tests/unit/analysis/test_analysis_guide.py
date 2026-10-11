"""Synthetic rendering/ownership tests; raw producer checks live in the oracle."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pyarrow as pa
import pytest
from tests.unit.analysis.test_structure_reporting import _cfg, _publish_inputs

from farkle.analysis.analysis_guide import (
    QUESTIONS,
    _published_json,
    inspect_product,
    render_guide_markdown,
    validate_guide,
)
from farkle.analysis.structure_reporting import run
from farkle.config import ArtifactScope
from farkle.utils.artifact_contract import (
    make_artifact_sidecar,
    sha256_file,
    sidecar_path,
    validate_artifact_sidecar,
)
from farkle.utils.artifacts import write_parquet_artifact_atomic
from farkle.utils.authenticated_contract import load_authenticated_sidecar
from farkle.utils.stage_completion import stage_done_path


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    cfg = _cfg(tmp_path_factory.mktemp("task8-synthetic"))
    _publish_inputs(cfg)
    run(cfg, execution_scope="single_root")
    return cfg, json.loads(cfg.analysis_guide_json_path().read_text(encoding="utf-8"))


def test_frozen_workbook_questions_notes_framing_and_paths(rendered):
    cfg, guide = rendered
    validate_guide(guide)
    mapped = [q["workbook_cells"][0] for g in guide["question_groups"] for q in g["questions"]]
    contract = Path(__file__).resolve().parents[3] / "docs/implementation/task0_question_map.md"
    expected = re.findall(r"^\| (A\d+) \|", contract.read_text(encoding="utf-8"), re.M)
    assert mapped == expected == [r[0] for r in QUESTIONS]
    assert len(mapped) == 62 and len(guide["question_groups"]) == 10
    assert set(guide["coverage"]["workbook"]["framing_cells"]) == {
        "A34",
        "A38",
        "A59",
        "A83",
        "A93",
    }
    assert (
        cfg.analysis_guide_json_path()
        == cfg.scope_dir("reporting", ArtifactScope.DIAGNOSTICS) / "analysis_guide.json"
    )
    assert cfg.analysis_guide_markdown_path() == cfg.analysis_guide_json_path().with_suffix(".md")


def test_actual_extraction_denominators_and_withheld_precision(rendered):
    cfg, guide = rendered
    products = {p["path"]: p for p in guide["artifact_dictionary"]}
    support = products[str(cfg.outcome_support_path(2))]
    assert support["examples"][0]["attempted"] == 4
    assert support["examples"][0]["completed"] == 3
    assert support["examples"][0]["safety"] == 1
    precision = products[str(cfg.outcome_precision_path(2))]
    assert precision["status_counts"] == {"status": {"insufficient_batches": 1}}
    assert precision["examples"][0]["estimate"] == 1 / 3
    assert precision["examples"][0]["denominator"] == 3
    assert precision["examples"][0]["ci_low"] is None
    assert precision["published_precision_ranges"]["denominator"] == [3, 3]
    assert "fixture_label" in products[str(cfg.rng_preflight_dir() / "publication.json")]["columns"]
    assert guide["run_identity"]["profile"]["production_eligible"] is False


def test_status_exclusions_and_claim_boundaries(rendered):
    _, guide = rendered
    questions = {
        q["workbook_cells"][0]: q for g in guide["question_groups"] for q in g["questions"]
    }
    for cell in ("A29", "A31", "A81"):
        assert questions[cell]["status"] == "deferred"
        assert not questions[cell]["results"]
    assert questions["A97"]["status"] == "excluded"
    for cell in ("A6", "A30", "A70"):
        assert questions[cell]["status"] == "partial"
    methods = " ".join(guide["limitations"])
    for phrase in (
        "not relative lift",
        "does not identify opponent-composition",
        "not an equivalence class",
        "does not promise decisive fronts",
        "comprehensive independence",
        "not causal",
    ):
        assert phrase in methods
    assert "sum(score)/sum(turns)" in questions["A61"]["denominator"]
    assert "Matched backgrounds" in questions["A89"]["denominator"]
    assert "B=30 or 100" in questions["A100"]["denominator"]
    assert guide["coverage"]["h2h"]["assumed_alternative"]["target_effect"] == 0.03
    pair = guide["coverage"]["h2h"]["supporting_pairwise_detail"][0]
    assert pair["simultaneous_interval_width"] == pytest.approx(
        pair["simultaneous_d_high"] - pair["simultaneous_d_low"]
    )


def test_json_rendering_preserves_values_and_h2h_presentation_order(rendered):
    cfg, guide = rendered
    markdown = cfg.analysis_guide_markdown_path().read_text(encoding="utf-8")
    assert markdown == render_guide_markdown(guide)
    assert f"(<{cfg.outcome_support_path(2).as_posix()}>)" in markdown
    assert (
        markdown.index("### Dominance fronts")
        < markdown.index("### Cycles")
        < markdown.index("### Unresolved and explicit pairwise equivalence")
        < markdown.index("Pairwise bounds, widths and decisions")
    )
    modified = copy.deepcopy(guide)
    modified["question_groups"][0]["questions"][0]["results"][0]["examples"][0][
        "chance_delta"
    ] = 0.123456789
    assert "0.123456789" in render_guide_markdown(modified)


def test_parameter_headline_retains_complete_matched_support_and_fold_evidence(rendered):
    cfg, guide = rendered
    questions = {
        q["workbook_cells"][0]: q for g in guide["question_groups"] for q in g["questions"]
    }
    headline = questions["A89"]["headline"][0]
    assert headline["exact_product_support"]["matched_support"]["artifact"] == str(
        cfg.parameter_response_support_path(2)
    )
    assert headline["exact_product_support"]["matched_support"]["status_counts"]
    assert headline["heldout_folds"]
    assert headline["permutation_association_importance"]


def test_authenticated_output_completion_and_unchanged_no_force_reuse(rendered):
    cfg, _ = rendered
    outputs = [cfg.analysis_guide_json_path(), cfg.analysis_guide_markdown_path()]
    for path in outputs:
        validate_artifact_sidecar(path, expected={"operation": "render_analysis_guide_v1"})
        sidecar = load_authenticated_sidecar(path)
        assert sidecar.versions.method_versions["analysis_guide_method_version"] == 1
        assert sidecar.versions.method_versions["structure_report_contract_version"] == 9
    md_identity = load_authenticated_sidecar(outputs[1])
    assert [s.artifact.location.relative_path for s in md_identity.source_artifacts] == [
        "analysis_guide.json"
    ]
    assert md_identity.source_artifacts[0].artifact.content_sha256 == sha256_file(outputs[0])
    assert (
        md_identity.source_artifacts[0].artifact == load_authenticated_sidecar(outputs[0]).artifact
    )
    done = json.loads(
        stage_done_path(cfg.stage_dir("reporting"), "structure_reporting").read_text()
    )
    assert {p.name for p in outputs} <= {
        o["artifact"]["location"]["relative_path"] for o in done["outputs"]
    }
    paths = [*outputs, *(sidecar_path(p) for p in outputs)]
    before = [(sha256_file(p), p.stat().st_mtime_ns) for p in paths]
    run(cfg, execution_scope="single_root")
    assert [(sha256_file(p), p.stat().st_mtime_ns) for p in paths] == before


@pytest.mark.parametrize("fault", ["question", "group", "notes", "columns", "result", "coverage"])
def test_contract_rejects_incomplete_or_invented_guide(rendered, fault):
    _, original = rendered
    guide = copy.deepcopy(original)
    question = guide["question_groups"][0]["questions"][0]
    if fault == "question":
        guide["question_groups"][0]["questions"].pop()
    elif fault == "group":
        guide["question_groups"].pop()
    elif fault == "notes":
        guide["methodological_notes"].pop()
    elif fault == "columns":
        question["evidence"][0]["columns"].append("unpublished_column")
    elif fault == "result":
        question["results"][0]["examples"][0]["chance_delta"] = 999
    else:
        question["status"] = "missing_required_data"
    with pytest.raises(ValueError):
        validate_guide(guide)


def test_missing_or_tampered_required_artifact_blocks_publication(rendered):
    cfg, _ = rendered
    path = cfg.outcome_support_path(2)
    original = path.read_bytes()
    output_hash = sha256_file(cfg.analysis_guide_json_path())
    try:
        path.unlink()
        with pytest.raises((RuntimeError, ValueError, FileNotFoundError)):
            run(cfg, execution_scope="single_root")
        path.write_bytes(original + b"corrupt")
        with pytest.raises((RuntimeError, ValueError)):
            run(cfg, execution_scope="single_root")
        assert sha256_file(cfg.analysis_guide_json_path()) == output_hash
    finally:
        path.write_bytes(original)


def test_verified_zero_and_unestimable_values_are_retained_not_imputed(tmp_path):
    cfg = _cfg(tmp_path)
    path = cfg.parameter_responses_path(2)
    table = pa.Table.from_pylist(
        [
            {"status": "verified_zero", "denominator": 0, "estimate": None},
            {"status": "unestimable_common_support", "denominator": 0, "estimate": None},
            {"status": "ok", "denominator": 10, "estimate": 0.0},
        ]
    )
    metadata = make_artifact_sidecar(
        cfg,
        path,
        producer="synthetic_task8",
        scope="by_k",
        source_scope="synthetic_fixture",
        operation="synthetic_guide_fixture",
        consistency_columns=table.schema.names,
        player_counts=[2],
        required_player_counts=[2],
        missing_cell_policy="fail",
    )
    write_parquet_artifact_atomic(table, path, sidecar=metadata)
    product = inspect_product(path, 1024 * 1024)
    assert product["examples"] == table.to_pylist()
    assert product["status_counts"]["status"] == {
        "ok": 1,
        "unestimable_common_support": 1,
        "verified_zero": 1,
    }


def test_nonmutating_paths(tmp_path):
    cfg = _cfg(tmp_path)
    paths = [cfg.analysis_guide_json_path(), cfg.analysis_guide_markdown_path()]
    assert all(not p.parent.exists() for p in paths)


def test_nonfinite_published_values_retain_sign_status_and_null_distinction(tmp_path, rendered):
    cfg = _cfg(tmp_path)
    path = cfg.parameter_responses_path(2)
    table = pa.Table.from_pylist(
        [
            {"status": "zero_reference_scale", "estimate": float("-inf"), "denominator": 4},
            {"status": "zero_reference_scale", "estimate": float("inf"), "denominator": 4},
            {"status": "undefined", "estimate": float("nan"), "denominator": 0},
            {"status": "unestimable", "estimate": None, "denominator": 0},
            {"status": "ok", "estimate": 0.0, "denominator": 4},
        ]
    )
    metadata = make_artifact_sidecar(
        cfg,
        path,
        producer="synthetic_task8",
        scope="by_k",
        source_scope="synthetic_fixture",
        operation="synthetic_guide_fixture",
        consistency_columns=table.schema.names,
        player_counts=[2],
        required_player_counts=[2],
        missing_cell_policy="fail",
    )
    write_parquet_artifact_atomic(table, path, sidecar=metadata)
    product = inspect_product(path, 1024 * 1024)
    assert [r["estimate"] for r in product["examples"]] == [
        {"published_nonfinite": "-Infinity"},
        {"published_nonfinite": "Infinity"},
        {"published_nonfinite": "NaN"},
        None,
        0.0,
    ]
    assert [r["status"] for r in product["examples"]] == table["status"].to_pylist()
    guide = copy.deepcopy(rendered[1])
    guide["coverage"]["published_representation_fixture"] = _published_json(
        {"value": float("-inf"), "denominator": 4, "status": "zero_reference_scale"}
    )
    persisted = json.loads(json.dumps(guide, allow_nan=False))
    assert render_guide_markdown(guide) == render_guide_markdown(persisted)
    assert '"published_nonfinite": "-Infinity"' in render_guide_markdown(persisted)


def test_missing_markdown_resume_preserves_valid_json_child(rendered):
    cfg, _ = rendered
    json_path = cfg.analysis_guide_json_path()
    md_path = cfg.analysis_guide_markdown_path()
    preserved = [json_path, sidecar_path(json_path)]
    before = [(sha256_file(p), p.stat().st_mtime_ns) for p in preserved]
    md_hash = sha256_file(md_path)
    md_path.unlink()
    run(cfg, execution_scope="single_root")
    assert [(sha256_file(p), p.stat().st_mtime_ns) for p in preserved] == before
    assert sha256_file(md_path) == md_hash
    validate_artifact_sidecar(md_path)
