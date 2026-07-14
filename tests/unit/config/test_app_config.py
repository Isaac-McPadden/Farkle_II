"""Tests for the ``farkle.config`` helpers."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import pytest
import yaml

from farkle.analysis.stage_registry import StageDefinition, StageLayout, StagePlacement
from farkle.config import (
    AppConfig,
    ArtifactScope,
    IOConfig,
    SimConfig,
    _annotation_contains,
    _coerce,
    _deep_merge,
    _normalize_seed_list,
    _normalize_seed_pair,
    apply_dot_overrides,
    expected_seed_list_length,
    load_app_config,
)
from farkle.utils.types import normalize_compression


@pytest.fixture
def write_yaml(tmp_path: Path):
    def _write(name: str, data: object) -> Path:
        path = tmp_path / name
        path.write_text(yaml.safe_dump(data))
        return path

    return _write


def test_load_app_config_merges_overlays(write_yaml) -> None:
    base = write_yaml(
        "base.yaml",
        {
            "io": {
                "results_dir_prefix": "results",
            },
            "analysis": {"log_level": "INFO"},
            "ingest": {"n_jobs": 2},
        },
    )
    overlay = write_yaml(
        "overlay.yaml",
        {
            "io.analysis_subdir": "custom",
            "analysis": {"outputs": {"metrics_name": "metrics.parquet"}},
            "ingest": {"batch_rows": 200_000},
        },
    )

    cfg = load_app_config(base, overlay)

    assert cfg.analysis.outputs["metrics_name"] == "metrics.parquet"
    assert cfg.ingest.batch_rows == 200_000
    # ensure base values remain when not overridden
    assert cfg.ingest.n_jobs == 2
    assert cfg.analysis_dir == Path("data") / "results_seed_0" / "custom"


def test_load_app_config_applies_analysis_controls(write_yaml) -> None:
    config = write_yaml(
        "analysis_fields.yaml",
        {
            "analysis": {
                "run_post_h2h_analysis": True,
                "run_frequentist": True,
                "run_agreement": True,
                "run_report": False,
                "head2head_target_hours": 4.5,
                "head2head_tolerance_pct": 2.5,
                "head2head_games_per_sec": 11.0,
                "frequentist_seeds": [3, 7],
            }
        },
    )

    cfg = load_app_config(config)

    assert cfg.analysis.run_post_h2h_analysis is True
    assert cfg.analysis.run_frequentist is True
    assert cfg.analysis.run_agreement is True
    assert cfg.analysis.run_report is False
    assert cfg.analysis.head2head_target_hours == pytest.approx(4.5)
    assert cfg.analysis.head2head_tolerance_pct == pytest.approx(2.5)
    assert cfg.analysis.head2head_games_per_sec == pytest.approx(11.0)
    assert cfg.analysis.frequentist_seeds == [3, 7]

    apply_dot_overrides(
        cfg,
        [
            "analysis.head2head_target_hours=1.25",
            "analysis.run_report=true",
            "analysis.run_agreement=false",
        ],
    )

    assert cfg.analysis.head2head_target_hours == pytest.approx(1.25)
    assert cfg.analysis.run_report is True
    assert cfg.analysis.run_agreement is False


def test_load_app_config_normalizes_legacy_keys(write_yaml) -> None:
    legacy = write_yaml(
        "legacy.yaml",
        {
            "io": {"analysis_dir": "analysis"},
            "sim": {"n_players": 5, "collect_metrics": True},
        },
    )

    cfg = load_app_config(legacy)

    assert cfg.io.analysis_subdir == "analysis"
    assert cfg.sim.n_players_list == [5]
    assert cfg.sim.expanded_metrics is True




def test_load_app_config_normalizes_legacy_keys_from_read_only_mappings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "legacy_read_only.yaml"
    config.write_text("{}")

    overlay = MappingProxyType(
        {
            "io": MappingProxyType({"analysis_dir": "analysis", "results_dir": "data/custom_seed_11"}),
            "sim": MappingProxyType({"n_players": 5, "collect_metrics": True}),
            "analysis": MappingProxyType({"run_frequentist": True}),
        }
    )

    monkeypatch.setattr("farkle.config.expand_dotted_keys", lambda _payload: overlay)

    cfg = load_app_config(config)

    assert cfg.io.analysis_subdir == "analysis"
    assert cfg.io.results_dir_prefix == Path("custom")
    assert cfg.sim.n_players_list == [5]
    assert cfg.sim.expanded_metrics is True
    assert cfg.analysis.run_frequentist is True


def test_load_app_config_handles_read_only_sim_without_mutating_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "sim_read_only.yaml"
    config.write_text("{}")
    sim_payload = MappingProxyType({"n_players": 7, "collect_metrics": True})

    monkeypatch.setattr("farkle.config.expand_dotted_keys", lambda _payload: {"sim": sim_payload})

    cfg = load_app_config(config)

    assert sim_payload == {"n_players": 7, "collect_metrics": True}
    assert cfg.sim.n_players_list == [7]
    assert cfg.sim.expanded_metrics is True

def test_load_app_config_keeps_results_dir(write_yaml) -> None:
    config = write_yaml(
        "seeded.yaml",
        {
            "io": {"results_dir_prefix": "base"},
            "sim": {"seed": 7},
        },
    )

    cfg = load_app_config(config)

    assert cfg.results_root == Path("data") / "base_seed_7"
    assert cfg.analysis_dir == Path("data") / "base_seed_7" / cfg.io.analysis_subdir


def test_load_app_config_rejects_non_mapping(tmp_path: Path) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text(yaml.safe_dump(["not", "a", "mapping"]))

    with pytest.raises(TypeError):
        load_app_config(config)


@pytest.mark.parametrize(
    ("payload", "error", "message"),
    [
        ({"sim": "bad-shape"}, TypeError, "Config section 'sim' must be a mapping"),
        (
            {"sim": {"per_n": [1, 2]}},
            TypeError,
            "sim.per_n must be a mapping of per-player overrides",
        ),
    ],
)
def test_load_app_config_rejects_invalid_section_shapes(
    write_yaml, payload, error, message
) -> None:
    config = write_yaml("invalid_shapes.yaml", payload)

    with pytest.raises(error, match=message):
        load_app_config(config)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {"simm": {"seed": 1}},
            "Unknown top-level config section\\(s\\): 'simm' \\(did you mean 'sim'\\?\\)",
        ),
        (
            {"sim": {"n_playerz": [5]}},
            "Unknown key\\(s\\) in config section 'sim': 'n_playerz' \\(did you mean 'n_players_list'\\?\\)",
        ),
        (
            {"sim": {"per_n": {5: {"n_jobz": 5}}}},
            "Unknown key\\(s\\) in config section sim.per_n\\[5\\]: 'n_jobz' \\(did you mean 'n_jobs'\\?\\)",
        ),
    ],
)
def test_load_app_config_reports_unknown_keys(write_yaml, payload, message: str) -> None:
    config = write_yaml("unknown_keys.yaml", payload)

    with pytest.raises(ValueError, match=message):
        load_app_config(config)


def test_apply_dot_overrides_coerces_values() -> None:
    cfg = AppConfig()

    apply_dot_overrides(
        cfg,
        [
            "io.results_dir_prefix=/tmp/output",
            "sim.seed=9",
            "trueskill.beta=32.5",
        ],
    )

    assert cfg.io.results_dir_prefix == Path("/tmp/output")
    assert cfg.sim.seed == 9
    assert cfg.trueskill.beta == pytest.approx(32.5)


@pytest.mark.parametrize(
    "override",
    ["missing_equals", "simseed=9"],
)
def test_apply_dot_overrides_rejects_bad_pairs(override: str) -> None:
    cfg = AppConfig()

    with pytest.raises(ValueError):
        apply_dot_overrides(cfg, [override])


def test_apply_dot_overrides_unknown_option() -> None:
    cfg = AppConfig()

    with pytest.raises(AttributeError):
        apply_dot_overrides(cfg, ["sim.unknown=1"])


@pytest.mark.parametrize(
    ("override", "attribute", "expected"),
    [
        ("analysis.run_report=ON", ("analysis", "run_report"), True),
        ("sim.seed=42", ("sim", "seed"), 42),
        ("io.results_dir_prefix=tmp/output_seed_9", ("io", "results_dir_prefix"), Path("tmp/output")),
        
    ],
)
def test_apply_dot_overrides_edge_case_coercions(override, attribute, expected) -> None:
    cfg = AppConfig()

    apply_dot_overrides(cfg, [override])

    section_name, option = attribute
    assert getattr(getattr(cfg, section_name), option) == expected




def test_apply_dot_overrides_list_coercion_edge_case_raises_value_error() -> None:
    cfg = AppConfig()

    with pytest.raises(ValueError, match=r"invalid literal for int\(\)"):
        apply_dot_overrides(cfg, ["sim.seed_list=1,2"])


def test_deep_merge_recursively_overrides_only_overlay_keys() -> None:
    merged = _deep_merge(
        {"sim": {"seed": 1, "per_n": {5: {"num_shuffles": 10}}}, "analysis": {"log_level": "INFO"}},
        {"sim": {"per_n": {5: {"seed": 3}}}, "analysis": {"run_report": False}},
    )

    assert merged == {
        "sim": {"seed": 1, "per_n": {5: {"num_shuffles": 10, "seed": 3}}},
        "analysis": {"log_level": "INFO", "run_report": False},
    }


@pytest.mark.parametrize(
    ("value", "current", "annotation", "expected"),
    [
        ("true", False, bool, True),
        ("11", 0, int, 11),
        ("5.5", 0.0, float, 5.5),
        ("tmp/results", Path("results"), Path, Path("tmp/results")),
    ],
)
def test_coerce_type_behavior(value, current, annotation, expected) -> None:
    assert _coerce(value, current, annotation) == expected


def test_coerce_list_annotation_edge_case_raises_value_error() -> None:
    with pytest.raises(ValueError, match=r"invalid literal for int\(\)"):
        _coerce("1,2,3", None, list[int])

def test_coerce_rejects_invalid_booleans() -> None:
    with pytest.raises(ValueError, match="Cannot parse boolean value"):
        _coerce("not_bool", False, bool)


@pytest.mark.parametrize(
    ("annotation", "target", "expected"),
    [
        (Path | None, Path, True),
        (list[int] | None, int, True),
        (tuple[str, ...], int, False),
        (None, bool, False),
    ],
)
def test_annotation_contains(annotation, target, expected: bool) -> None:
    assert _annotation_contains(annotation, target) is expected


@pytest.mark.parametrize(
    ("payload", "seed_list_len", "message"),
    [
        (
            {"sim": {"seed": 9, "seed_pair": [8, 9]}},
            None,
            "sim.seed must match seed_pair\\[0\\] when both are set",
        ),
        (
            {"sim": {"seed_list": [1, 2], "seed_pair": [1, 3]}},
            None,
            "load_app_config: sim.seed_list and sim.seed_pair must match when both are set",
        ),
        (
            {"sim": {"seed_list": [1, 2]}},
            1,
            "load_app_config: sim.seed_list must contain exactly 1 seeds, got \\[1, 2\\]",
        ),
    ],
)
def test_load_app_config_seed_source_conflicts(write_yaml, payload, seed_list_len, message) -> None:
    config = write_yaml("seed_conflicts.yaml", payload)

    with pytest.raises(ValueError, match=message):
        load_app_config(config, seed_list_len=seed_list_len)


def test_load_app_config_seed_list_normalization(write_yaml) -> None:
    config = write_yaml("seed_normalization.yaml", {"sim": {"seed_list": ("8", "9")}})

    cfg = load_app_config(config, seed_list_len=2)

    assert cfg.sim.seed_list == [8, 9]
    assert cfg.sim.seed == 8
    assert cfg.sim.seed_pair == (8, 9)


# --- seed validation + normalization branches ---


@pytest.mark.parametrize(
    ("seed_list", "error", "message"),
    [
        (123, TypeError, "sim.seed_list must be a list/tuple of integers"),
        ([], ValueError, "sim.seed_list must contain at least one seed"),
    ],
)
def test_normalize_seed_list_rejects_invalid_shapes(seed_list, error, message: str) -> None:
    sim = SimConfig(seed_list=seed_list)

    with pytest.raises(error, match=message):
        _normalize_seed_list(sim)


@pytest.mark.parametrize(
    ("seed_pair", "seed_provided", "error", "message"),
    [
        (123, False, TypeError, "sim.seed_pair must be a tuple/list of two integers"),
        ([1], False, ValueError, "sim.seed_pair must contain exactly two seeds"),
    ],
)
def test_normalize_seed_pair_rejects_invalid_shapes(
    seed_pair, seed_provided: bool, error, message: str
) -> None:
    sim = SimConfig(seed_pair=seed_pair)

    with pytest.raises(error, match=message):
        _normalize_seed_pair(sim, seed_provided=seed_provided)


def test_normalize_seed_pair_autosyncs_seed_when_seed_not_explicitly_provided() -> None:
    sim = SimConfig(seed=999, seed_pair=(12, 34))

    _normalize_seed_pair(sim, seed_provided=False)

    assert sim.seed == 12


def test_load_app_config_seed_list_len_two_autopopulates_seed_pair(write_yaml) -> None:
    cfg = load_app_config(write_yaml("seed_pair_autopop.yaml", {"sim": {"seed_list": [31, 47]}}))

    assert cfg.sim.seed_pair == (31, 47)


def test_load_app_config_logs_legacy_seed_precedence_warning(
    write_yaml, caplog: pytest.LogCaptureFixture
) -> None:
    config = write_yaml(
        "seed_precedence_warning.yaml",
        {"sim": {"seed": 11, "seed_pair": [11, 12], "seed_list": [11, 12]}},
    )

    with caplog.at_level("WARNING"):
        cfg = load_app_config(config)

    assert cfg.sim.seed == 11
    assert any(
        "sim.seed_list overrides legacy sim.seed/seed_pair settings" in rec.message
        for rec in caplog.records
    )


def test_load_app_config_per_n_seed_override_tracking_valid_int_key(write_yaml) -> None:
    config = write_yaml(
        "per_n_seed_tracking.yaml",
        {"sim": {"per_n": {"5": {"seed_list": [7, 8]}}}},
    )

    cfg = load_app_config(config)

    assert cfg.sim.per_n[5].seed_list == [7, 8]
    assert cfg.sim.per_n[5].seed == 7
    assert cfg.sim.per_n[5].seed_pair == (7, 8)


def test_load_app_config_per_n_non_int_key_raises_gracefully(write_yaml) -> None:
    config = write_yaml(
        "per_n_non_int_key.yaml",
        {"sim": {"per_n": {"not-an-int": {"seed": 100}}}},
    )

    with pytest.raises(ValueError, match="invalid literal for int"):
        load_app_config(config)


# --- path resolution + fallback order branches ---


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        (Path("/tmp/meta_abs"), Path("/tmp/meta_abs")),
        (Path("data/shared_meta"), Path("data/shared_meta")),
        (Path("shared/meta"), Path("data") / "shared" / "meta"),
    ],
)
def test_meta_analysis_dir_resolution_modes(configured: Path, expected: Path) -> None:
    cfg = AppConfig(io=IOConfig(meta_analysis_dir=configured))
    assert cfg.meta_analysis_dir == expected


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        (Path("/tmp/interseed_abs"), Path("/tmp/interseed_abs")),
        (Path("data/interseed_anchor"), Path("data/interseed_anchor")),
        (Path("upstream"), Path("data") / "results_seed_0" / "upstream"),
    ],
)
def test_interseed_input_dir_resolution_modes(configured: Path, expected: Path) -> None:
    cfg = AppConfig(io=IOConfig(interseed_input_dir=configured))
    assert cfg.interseed_input_dir == expected


def test_metrics_input_path_requires_canonical_scope(tmp_path: Path) -> None:
    cfg = AppConfig(
        io=IOConfig(
            results_dir_prefix=tmp_path / "results", interseed_input_dir=tmp_path / "upstream"
        )
    )
    name = "metrics.parquet"
    canonical = cfg.input_scope_path("metrics", ArtifactScope.ACROSS_K, name)
    generic_path = cfg._input_stage_path("metrics", "combined") / name  # type: ignore[operator]
    legacy_path = cfg.analysis_dir / name

    generic_path.parent.mkdir(parents=True, exist_ok=True)
    generic_path.write_text("generic")
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_text("legacy")
    assert cfg.metrics_input_path(name) == canonical
    assert cfg.metrics_input_path(name) not in {generic_path, legacy_path}


def test_meta_input_path_requires_by_k_scope(tmp_path: Path) -> None:
    cfg = AppConfig(
        io=IOConfig(
            results_dir_prefix=tmp_path / "results", interseed_input_dir=tmp_path / "upstream"
        )
    )
    name = "meta.json"
    expected = cfg.input_scope_path("meta", ArtifactScope.BY_K, name, k=5)
    wrong_scope = cfg.input_scope_path("meta", ArtifactScope.CROSS_SEED, name)
    wrong_scope.parent.mkdir(parents=True, exist_ok=True)
    wrong_scope.write_text("wrong")

    assert cfg.meta_input_path(5, name) == expected
    assert cfg.meta_input_path(5, name) != wrong_scope


def test_post_h2h_path_fallback_ordering(tmp_path: Path) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "results"))
    filename = "ratings_combined.parquet"
    canonical = cfg.canonical_artifact_name(filename)
    post_h2h_stage = cfg._stage_dir_if_active("post_h2h") / canonical  # type: ignore[operator]
    head2head_stage = cfg._stage_dir_if_active("head2head") / canonical  # type: ignore[operator]
    legacy = cfg.analysis_dir / canonical

    post_h2h_stage.parent.mkdir(parents=True, exist_ok=True)
    post_h2h_stage.write_text("post_h2h")
    assert cfg.post_h2h_path(filename) == post_h2h_stage

    post_h2h_stage.unlink()
    head2head_stage.parent.mkdir(parents=True, exist_ok=True)
    head2head_stage.write_text("head2head")
    assert cfg.post_h2h_path(filename) == head2head_stage

    head2head_stage.unlink()
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text("legacy")
    assert cfg.post_h2h_path(filename) == legacy


# --- override coercion + warnings ---


def test_apply_dot_overrides_alias_remap_and_deprecated_flag_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = AppConfig()

    with caplog.at_level("WARNING"):
        apply_dot_overrides(
            cfg, ["io.results_dir=/tmp/remapped_seed_9", "analysis.run_report=true"]
        )

    assert cfg.io.results_dir_prefix == Path("/tmp/remapped")
    assert any(
        "Deprecated analysis flag override provided; stages ignore it" in rec.message
        for rec in caplog.records
    )


def test_apply_dot_overrides_invalid_bool_raises() -> None:
    cfg = AppConfig()

    with pytest.raises(ValueError, match="Cannot parse boolean value"):
        apply_dot_overrides(cfg, ["analysis.run_report=definitely-not-bool"])


def test_load_app_config_overlay_and_dot_override_round_trip_is_deterministic(write_yaml) -> None:
    base = write_yaml(
        "round_trip_base.yaml",
        {
            "io": {"results_dir_prefix": "results"},
            "sim": {"seed": 3, "n_players_list": [5, 7]},
            "analysis": {"log_level": "INFO", "run_report": False},
        },
    )
    overlay = write_yaml(
        "round_trip_overlay.yaml",
        {
            "io.analysis_subdir": "custom",
            "sim": {"seed_list": [3]},
            "screening": {"resolution_delta": 0.04},
            "analysis": {"run_report": True},
        },
    )
    overrides = [
        "analysis.log_level=DEBUG",
        "sim.seed=3",
        "io.results_dir_prefix=data/results_seed_3",
    ]

    cfg_a = apply_dot_overrides(load_app_config(base, overlay), overrides)
    cfg_b = apply_dot_overrides(load_app_config(base, overlay), overrides)

    assert cfg_a == cfg_b
    assert cfg_a.analysis_dir == Path("data") / "results_seed_3" / "custom"


def test_curated_parquet_falls_back_without_curate_stage(tmp_path: Path) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path))
    cfg.set_stage_layout(
        StageLayout(
            placements=[
                StagePlacement(
                    definition=StageDefinition(key="combine", group="pipeline"),
                    index=0,
                    folder_name="00_combine",
                )
            ]
        )
    )

    curated = cfg.curated_parquet

    assert curated.name == "all_ingested_rows.parquet"
    assert cfg.data_dir == cfg.analysis_dir / "curate"


def test_curated_parquet_prefers_interseed_combine_input(tmp_path: Path) -> None:
    upstream_root = tmp_path / "upstream"
    combine_folder = "02_combine"
    upstream_curated = upstream_root / combine_folder / "concat_ks" / "all_ingested_rows.parquet"
    upstream_curated.parent.mkdir(parents=True, exist_ok=True)
    upstream_curated.write_text("rows")

    cfg = AppConfig(
        io=IOConfig(
            results_dir_prefix=tmp_path / "pair_results",
            interseed_input_dir=upstream_root,
            interseed_input_layout={"combine": combine_folder},
        )
    )

    curated = cfg.curated_parquet

    assert curated == upstream_curated
    assert cfg.curated_parquet_candidates()[0] == upstream_curated


def test_curate_stage_dir_prefers_layout_folder(tmp_path: Path) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path))
    layout = cfg.stage_layout

    cfg.set_stage_layout(layout)

    folder = layout.folder_for("curate")
    assert folder is not None
    assert cfg.curate_stage_dir == cfg.analysis_dir / folder


@pytest.mark.parametrize(
    ("payload", "expected_results_root", "expected_analysis"),
    [
        (
            {"io": {"results_dir_prefix": "results"}, "sim": {"seed": 3}},
            Path("data") / "results_seed_3",
            Path("data") / "results_seed_3" / "analysis",
        ),
        (
            {
                "io": {
                    "results_dir": "data/custom_seed_4",
                    "analysis_dir": "alt_analysis",
                }
            },
            Path("data") / "custom_seed_0",
            Path("data") / "custom_seed_0" / "alt_analysis",
        ),
    ],
)
def test_load_app_config_defaults_and_overrides(
    write_yaml, payload, expected_results_root, expected_analysis
) -> None:
    config = write_yaml("defaults_overrides.yaml", payload)

    cfg = load_app_config(config)

    assert cfg.results_root == expected_results_root
    assert cfg.analysis_dir == expected_analysis


@pytest.mark.parametrize(
    ("input_prefix", "expected_prefix"),
    [
        ("data/results_seed_10", Path("results")),
        ("nested/output_seed_2", Path("nested/output")),
        (Path("already_normal"), Path("already_normal")),
    ],
)
def test_results_dir_normalization_via_load_and_overrides(
    write_yaml, input_prefix, expected_prefix
) -> None:
    config = write_yaml("norm.yaml", {"io": {"results_dir": str(input_prefix)}})

    cfg = load_app_config(config)
    assert cfg.io.results_dir_prefix == expected_prefix

    apply_dot_overrides(cfg, [f"io.results_dir_prefix={input_prefix}"])
    assert cfg.io.results_dir_prefix == expected_prefix


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("analysis", "log_level", "NOT_A_LEVEL"),
        ("ingest", "parquet_codec", "zipper"),
        ("k_aggregation", "method", "mystery-weight"),
    ],
)
def test_invalid_enum_codec_and_compression_values(write_yaml, section, key, value) -> None:
    config = write_yaml("invalid_values.yaml", {section: {key: value}})

    cfg = load_app_config(config)

    if key == "parquet_codec":
        with pytest.raises(ValueError):
            _ = cfg.parquet_codec
    else:
        assert getattr(getattr(cfg, section), key) == value


@pytest.mark.parametrize(
    ("cfg", "expected"),
    [
        (AppConfig(), (False, "interseed inputs missing")),
        (AppConfig(sim=AppConfig().sim.__class__()), (False, "interseed inputs missing")),
    ],
)
def test_interseed_ready_missing_inputs(cfg: AppConfig, expected: tuple[bool, str]) -> None:
    ready, message = cfg.interseed_ready()

    assert ready is expected[0]
    assert expected[1] in message


@pytest.mark.parametrize(
    "cfg",
    [
        AppConfig(sim=AppConfig().sim.__class__(seed_list=[1, 2])),
        AppConfig(io=IOConfig(interseed_input_dir=Path("inputs"))),
    ],
)
def test_interseed_ready_accepts_seed_list_or_input_dir(cfg: AppConfig) -> None:
    ready, message = cfg.interseed_ready()

    assert ready is True
    assert message == ""


def test_stage_layout_edge_case_resolve_stage_dir_without_folder(tmp_path: Path) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path))
    cfg.set_stage_layout(
        StageLayout(
            placements=[
                StagePlacement(
                    definition=StageDefinition(key="ingest", group="pipeline"),
                    index=0,
                    folder_name="00_ingest",
                )
            ]
        )
    )

    with pytest.raises(KeyError):
        cfg.stage_dir("curate")

    assert cfg.resolve_stage_dir("curate", allow_missing=True) == cfg.analysis_dir / "curate"


def test_normalize_compression_rejects_invalid_codec() -> None:
    with pytest.raises(ValueError):
        normalize_compression("bad-codec")


@pytest.mark.parametrize(
    ("command", "subcommand", "expected"),
    [
        ("two-seed-pipeline", None, 2),
        ("run", None, 1),
        ("unknown", None, None),
    ],
)
def test_expected_seed_list_length(
    command: str, subcommand: str | None, expected: int | None
) -> None:
    assert expected_seed_list_length(command, subcommand=subcommand) == expected


def test_sim_config_resolve_seed_list_rejects_invalid_expected_len() -> None:
    with pytest.raises(ValueError, match="expected_len must be >= 1"):
        SimConfig().resolve_seed_list(0)


def test_sim_config_resolve_seed_list_rejects_seed_list_mismatch() -> None:
    sim = SimConfig(seed_list=[1, 2])

    with pytest.raises(ValueError, match=r"must contain exactly 1 seeds"):
        sim.resolve_seed_list(1)


def test_sim_config_resolve_seed_list_uses_seed_fallback_for_single_seed() -> None:
    assert SimConfig(seed=17).resolve_seed_list(1) == [17]


def test_sim_config_resolve_seed_list_uses_seed_pair_fallback_for_two_seed() -> None:
    assert SimConfig(seed_pair=(17, 23)).resolve_seed_list(2) == [17, 23]


def test_sim_config_resolve_seed_list_rejects_unsupported_length() -> None:
    with pytest.raises(ValueError, match="Unsupported expected seed length 3"):
        SimConfig().resolve_seed_list(3)


def test_sim_config_populate_seed_list_rejects_conflicting_sources() -> None:
    sim = SimConfig(seed_list=[1, 2], seed_pair=(1, 3))

    with pytest.raises(ValueError, match="sim.seed_list and sim.seed_pair must match"):
        sim.populate_seed_list(2)


def test_sim_config_populate_seed_list_expected_len_1_updates_seed() -> None:
    sim = SimConfig(seed=5, seed_list=[13])

    seeds = sim.populate_seed_list(1)

    assert seeds == [13]
    assert sim.seed == 13
    assert sim.seed_list == [13]


def test_sim_config_populate_seed_list_expected_len_2_sets_seed_pair_and_syncs_seed() -> None:
    sim = SimConfig(seed=5, seed_list=[13, 21])

    seeds = sim.populate_seed_list(2)

    assert seeds == [13, 21]
    assert sim.seed_pair == (13, 21)
    assert sim.seed == 13


def test_sim_config_require_seed_pair_rejects_invalid_seed_list_length() -> None:
    with pytest.raises(ValueError, match="must contain exactly two seeds"):
        SimConfig(seed_list=[1]).require_seed_pair()


def test_sim_config_require_seed_pair_rejects_missing_seed_pair() -> None:
    with pytest.raises(ValueError, match="sim.seed_pair must be set"):
        SimConfig().require_seed_pair()


def test_sim_config_require_seed_pair_accepts_seed_list_or_seed_pair() -> None:
    assert SimConfig(seed_list=[4, 9]).require_seed_pair() == (4, 9)
    assert SimConfig(seed_pair=(11, 12)).require_seed_pair() == (11, 12)


def test_interseed_input_folder_supports_none_mapping_and_stage_layout() -> None:
    cfg = AppConfig()
    assert cfg._interseed_input_folder("combine") is None

    cfg = AppConfig(io=IOConfig(interseed_input_layout={"combine": "02_combine"}))
    assert cfg._interseed_input_folder("combine") == "02_combine"

    layout = StageLayout(
        placements=[
            StagePlacement(
                definition=StageDefinition(key="combine", group="pipeline"),
                index=0,
                folder_name="02_combine",
            )
        ]
    )
    cfg = AppConfig(io=IOConfig(interseed_input_layout=layout))
    assert cfg._interseed_input_folder("combine") == "02_combine"


def test_resolve_input_stage_dir_prefers_input_root_then_stage_then_none(tmp_path: Path) -> None:
    cfg = AppConfig(
        io=IOConfig(
            interseed_input_dir=tmp_path / "upstream",
            interseed_input_layout={"combine": "02_combine"},
        )
    )
    assert (
        cfg.resolve_input_stage_dir("combine", "combined")
        == tmp_path / "upstream" / "02_combine" / "combined"
    )

    local_cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "results"))
    local_path = local_cfg.resolve_input_stage_dir("combine", "combined")
    assert local_path is not None
    combine_folder = local_cfg.stage_layout.folder_for("combine")
    assert combine_folder is not None
    assert local_path == local_cfg.analysis_dir / combine_folder / "combined"

    assert local_cfg.resolve_input_stage_dir("not_a_stage") is None


def test_stage_key_for_folder_matching_and_non_matching(tmp_path: Path) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path))
    combine_folder = cfg.stage_layout.folder_for("combine")
    assert combine_folder is not None

    assert cfg._stage_key_for_folder(combine_folder) == "combine"
    assert cfg._stage_key_for_folder("99_missing") is None


def test_interseed_input_candidate_relative_and_fallback(tmp_path: Path) -> None:
    cfg = AppConfig(
        io=IOConfig(
            results_dir_prefix=tmp_path / "results",
            interseed_input_dir=tmp_path / "upstream",
            interseed_input_layout={"combine": "02_combine"},
        )
    )
    combine_folder = cfg.stage_layout.folder_for("combine")
    assert combine_folder is not None

    inside_stage = cfg.analysis_dir / combine_folder / "combined"
    assert (
        cfg._interseed_input_candidate(inside_stage, "rows.parquet")
        == tmp_path / "upstream" / "02_combine" / "combined" / "rows.parquet"
    )

    external = tmp_path / "external"
    assert (
        cfg._interseed_input_candidate(external, "rows.parquet")
        == tmp_path / "upstream" / "rows.parquet"
    )


def test_preferred_stage_path_candidate_ordering(tmp_path: Path) -> None:
    cfg = AppConfig(
        io=IOConfig(
            results_dir_prefix=tmp_path / "results",
            interseed_input_dir=tmp_path / "upstream",
            interseed_input_layout={"trueskill": "05_trueskill"},
        )
    )
    stage_dir = cfg.analysis_dir / (cfg.stage_layout.folder_for("trueskill") or "trueskill")
    legacy_dir = cfg.analysis_dir
    stage_dir.mkdir(parents=True, exist_ok=True)
    filename = "ratings_k_weighted.parquet"

    stage_path = stage_dir / filename
    stage_path.write_text("stage")
    assert cfg._preferred_stage_path(stage_dir, legacy_dir, filename) == stage_path

    stage_path.unlink()
    interseed_path = tmp_path / "upstream" / "05_trueskill" / filename
    interseed_path.parent.mkdir(parents=True, exist_ok=True)
    interseed_path.write_text("input")
    assert cfg._preferred_stage_path(stage_dir, legacy_dir, filename) == interseed_path

    interseed_path.unlink()
    legacy_path = legacy_dir / filename
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_text("legacy")
    assert cfg._preferred_stage_path(stage_dir, legacy_dir, filename) == legacy_path

    legacy_path.unlink()
    assert cfg._preferred_stage_path(stage_dir, legacy_dir, filename) == stage_path


def test_resolve_stage_artifact_path_ordering_for_combine_and_non_combine(tmp_path: Path) -> None:
    cfg = AppConfig(
        io=IOConfig(
            results_dir_prefix=tmp_path / "results",
            interseed_input_dir=tmp_path / "upstream",
            interseed_input_layout={"combine": "01_combine", "screening": "08_screening"},
        )
    )

    combine_stage = cfg._stage_dir_if_active("combine", "combined")
    assert combine_stage is not None
    combine_local = combine_stage / "all_ingested_rows.parquet"
    combine_local.parent.mkdir(parents=True, exist_ok=True)
    combine_local.write_text("local")
    combine_input = tmp_path / "upstream" / "01_combine" / "combined" / "all_ingested_rows.parquet"
    combine_input.parent.mkdir(parents=True, exist_ok=True)
    combine_input.write_text("input")
    assert (
        cfg._resolve_stage_artifact_path("combine", "all_ingested_rows.parquet", "combined")
        == combine_input
    )

    screening_stage = cfg._stage_dir_if_active("screening")
    assert screening_stage is not None
    screening_local = screening_stage / "descriptive_screening.json"
    screening_local.parent.mkdir(parents=True, exist_ok=True)
    screening_local.write_text("local")
    screening_input = tmp_path / "upstream" / "08_screening" / "descriptive_screening.json"
    screening_input.parent.mkdir(parents=True, exist_ok=True)
    screening_input.write_text("input")
    assert (
        cfg._resolve_stage_artifact_path("screening", "descriptive_screening.json")
        == screening_local
    )

    screening_local.unlink()
    screening_input.unlink()
    legacy = cfg.analysis_dir / "legacy_screening.json"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text("legacy")
    assert (
        cfg._resolve_stage_artifact_path(
            "screening", "descriptive_screening.json", legacy_paths=(legacy,)
        )
        == legacy
    )


def test_preferred_tiers_path_fallback_ordering(tmp_path: Path) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "results"))

    frequentist_path = cfg._resolve_stage_artifact_path("frequentist", "tiers.json")
    frequentist_path.parent.mkdir(parents=True, exist_ok=True)
    frequentist_path.write_text("frequentist")
    assert cfg.preferred_tiers_path() == frequentist_path

    frequentist_path.unlink()
    trueskill_path = cfg._resolve_stage_artifact_path("trueskill", "tiers.json")
    trueskill_path.parent.mkdir(parents=True, exist_ok=True)
    trueskill_path.write_text("trueskill")
    assert cfg.preferred_tiers_path() == trueskill_path

    trueskill_path.unlink()
    analysis_path = cfg.analysis_dir / "tiers.json"
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_path.write_text("analysis")
    assert cfg.preferred_tiers_path() == analysis_path

    analysis_path.unlink()
    assert cfg.preferred_tiers_path() == cfg._resolve_stage_artifact_path(
        "frequentist", "tiers.json"
    )


def test_load_app_config_rejects_all_k_player_sentinels(write_yaml) -> None:
    text_sentinel = write_yaml("text_sentinel.yaml", {"sim": {"n_players_list": ["combined"]}})
    numeric_sentinel = write_yaml("numeric_sentinel.yaml", {"sim": {"n_players_list": [0]}})

    with pytest.raises(ValueError, match="concrete player counts"):
        load_app_config(text_sentinel)
    with pytest.raises(ValueError, match="concrete player counts"):
        load_app_config(numeric_sentinel)

    bad_config = write_yaml("bad_players.yaml", {"sim": {"n_players_list": ["abc"]}})
    with pytest.raises(ValueError, match="invalid n_players_list entry"):
        load_app_config(bad_config)


def test_load_app_config_reads_run_frequentist(write_yaml) -> None:
    config = write_yaml("freq_flag.yaml", {"analysis": {"run_frequentist": False}})

    cfg = load_app_config(config)

    assert cfg.analysis.run_frequentist is False


def test_load_app_config_rejects_non_sim_bad_section_shape(write_yaml) -> None:
    config = write_yaml("bad_analysis_shape.yaml", {"analysis": "bad"})

    with pytest.raises(TypeError, match="Config section 'analysis' must be a mapping"):
        load_app_config(config)


def test_small_properties_cover_combined_and_trueskill_variants(tmp_path: Path) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "results"))

    across_k_path = cfg.agreement_across_k_output_path()
    assert across_k_path.name == "agreement_across_k.json"
    assert across_k_path.parent.name == "diagnostics"
    assert cfg.agreement_output_path(5).name == "agreement_5p.json"

    combined_trueskill = cfg.trueskill_path("ratings_combined.parquet")
    assert combined_trueskill.name in {"ratings_k_weighted.parquet", "ratings_combined.parquet"}
    non_combined = cfg.trueskill_path("tiers.json")
    assert non_combined.name == "tiers.json"


def test_app_config_stage_and_alias_helpers_cover_common_paths(tmp_path: Path) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "results"))

    assert cfg.ingest_block_dir(5).name == "5p"
    assert cfg.curate_block_dir(5).name == "5p"
    assert cfg.combine_block_dir(5).name == "5p"
    assert cfg.combine_combined_dir().name == "concat_ks"
    assert cfg.metrics_per_k_dir(5).name == "5p"
    assert cfg.metrics_combined_dir.name == "across_k"

    assert cfg.game_stats_stage_dir.parent == cfg.analysis_dir
    assert cfg.game_stats_combined_dir.name == "across_k"
    assert cfg.seed_summaries_stage_dir.parent == cfg.analysis_dir
    assert cfg.seed_summaries_dir(5).name == "5p"
    assert cfg.variance_stage_dir.parent == cfg.analysis_dir
    assert cfg.variance_combined_dir.name == "cross_seed"
    assert cfg.meta_stage_dir.parent == cfg.analysis_dir
    assert cfg.meta_combined_dir.name == "cross_seed"
    assert cfg.agreement_stage_dir.parent == cfg.analysis_dir
    assert cfg.interseed_stage_dir.parent == cfg.analysis_dir
    assert cfg.ingest_stage_dir.parent == cfg.analysis_dir
    assert cfg.combine_stage_dir.parent == cfg.analysis_dir
    assert cfg.metrics_stage_dir.parent == cfg.analysis_dir
    assert cfg.trueskill_stage_dir.parent == cfg.analysis_dir
    assert cfg.trueskill_combined_dir.name == "across_k"
    assert cfg.head2head_stage_dir.parent == cfg.analysis_dir
    assert cfg.post_h2h_stage_dir.parent == cfg.analysis_dir
    assert cfg.hgb_stage_dir.parent == cfg.analysis_dir
    assert cfg.hgb_per_k_dir(7).name == "7p"
    assert cfg.hgb_combined_dir.name == "across_k"
    assert cfg.frequentist_stage_dir.parent == cfg.analysis_dir

    assert cfg.n_dir(5).name == "5_players"
    assert cfg.checkpoint_path(5).name == "5p_checkpoint.pkl"
    assert cfg.metrics_path(5).name == "5p_metrics.parquet"
    assert cfg.strategy_manifest_path(5) == cfg.strategy_manifest_root_path()
    assert cfg.row_group_size == cfg.ingest.row_group_size
    assert cfg.n_jobs_ingest == cfg.ingest.n_jobs
    assert cfg.batch_rows == cfg.ingest.batch_rows
    assert cfg.trueskill_beta == cfg.trueskill.beta
    assert cfg.hgb_max_iter == cfg.hgb.n_estimators
    assert cfg.combine_max_players == cfg.combine.max_players
    assert cfg.metrics_seat_range == cfg.metrics.seat_range


def test_app_config_input_output_helpers_cover_stage_wrappers(tmp_path: Path) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "results"))

    assert cfg.metrics_output_path("m.parquet").name == "m.parquet"
    assert (
        cfg.game_stats_output_path("margin_combined.parquet").name
        == "margin_strategy_conditioned_equal_k_mean.parquet"
    )
    assert cfg.game_stats_input_path("stats.parquet").name == "stats.parquet"
    assert cfg.variance_output_path("variance.json").name == "variance.json"
    assert cfg.variance_input_path("variance.json").name == "variance.json"
    assert cfg.meta_output_path(5, "meta.json").name == "meta.json"


def test_metrics_input_path_defaults_to_interseed_then_stage_when_missing(tmp_path: Path) -> None:
    cfg = AppConfig(
        io=IOConfig(
            results_dir_prefix=tmp_path / "results", interseed_input_dir=tmp_path / "upstream"
        )
    )

    interseed_default = cfg._input_stage_path("metrics", "across_k")
    assert interseed_default is not None
    assert cfg.metrics_input_path() == interseed_default / cfg.metrics_name

    cfg_no_input = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "local_only"))
    stage_default = cfg_no_input._stage_dir_if_active("metrics", "across_k")
    assert stage_default is not None
    assert cfg_no_input.metrics_input_path() == stage_default / cfg_no_input.metrics_name


def test_additional_branch_coverage_for_optional_stage_and_input_helpers(tmp_path: Path) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "results"))

    assert cfg.meta_analysis_dir == cfg.analysis_dir
    assert cfg._interseed_input_folder(None) is None
    assert cfg._interseed_input_candidate(cfg.analysis_dir, "x.txt") is None
    assert cfg._stage_dir_if_active("not_registered") is None
    assert cfg._interseed_stage_dir("combine") is None

    cfg.io.interseed_input_layout = cast(Any, object())
    assert cfg._interseed_input_folder("combine") is None


def test_rng_paths_when_rng_stage_registered(tmp_path: Path) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "results"))
    cfg.set_stage_layout(
        StageLayout(
            placements=[
                StagePlacement(
                    definition=StageDefinition(key="rng_diagnostics", group="analysis"),
                    index=0,
                    folder_name="00_rng",
                )
            ]
        )
    )

    out_path = cfg.rng_output_path("rng.json")
    out_path.write_text("rng")

    assert cfg.rng_combined_dir.name == "diagnostics"
    assert cfg.rng_stage_dir.parent == cfg.analysis_dir
    assert cfg.rng_input_path("rng.json") == out_path


def test_canonical_scope_helpers_are_disjoint_and_validate_k(tmp_path: Path) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "results"))

    roots = {
        cfg.by_k_dir("metrics", 4),
        cfg.concat_ks_dir("metrics"),
        cfg.across_k_dir("metrics"),
        cfg.cross_seed_dir("metrics"),
        cfg.diagnostics_dir("metrics"),
        cfg.h2h_2p_dir("metrics"),
    }
    assert len(roots) == len(ArtifactScope)
    assert cfg.by_k_dir("metrics", 4).parts[-2:] == ("by_k", "4p")

    with pytest.raises(ValueError, match="requires a concrete positive player count"):
        cfg.scope_dir("metrics", ArtifactScope.BY_K)
    with pytest.raises(ValueError, match="does not accept a player count"):
        cfg.scope_dir("metrics", ArtifactScope.ACROSS_K, k=4)


def test_scope_guard_rejects_stage_scope_and_k_interchange(tmp_path: Path) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "results"))
    per_k = cfg.scope_path("metrics", ArtifactScope.BY_K, "metrics.parquet", k=4)
    across_k = cfg.scope_path("metrics", ArtifactScope.ACROSS_K, "metrics.parquet")
    other_root = cfg.scope_path("trueskill", ArtifactScope.BY_K, "ratings.parquet", k=4)

    assert (
        cfg.require_scope(
            per_k,
            stage="metrics",
            scope=ArtifactScope.BY_K,
            k=4,
        )
        == per_k
    )
    for wrong_path, stage, scope, k in (
        (across_k, "metrics", ArtifactScope.BY_K, 4),
        (per_k, "metrics", ArtifactScope.BY_K, 2),
        (other_root, "metrics", ArtifactScope.BY_K, 4),
    ):
        with pytest.raises(ValueError, match="does not belong to required scope"):
            cfg.require_scope(wrong_path, stage=stage, scope=scope, k=k)


def test_statistical_contract_accepts_complete_locked_configuration() -> None:
    cfg = AppConfig(sim=SimConfig(n_players_list=[2, 4], seed_list=[101, 202]))
    cfg.screening.practical_delta_by_k = {2: 0.03, 4: 0.015}
    cfg.screening.delta_across_k = 0.02

    cfg.validate_statistical_contract(require_two_roots=True)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("max_shuffles_per_root_k", 0, "max_shuffles_per_root_k"),
        ("projected_games_per_second", 0.0, "projected_games_per_second"),
        ("projected_games_per_second", float("nan"), "projected_games_per_second"),
    ],
)
def test_statistical_contract_validates_screening_workload_limits(
    field: str,
    value: int | float,
    message: str,
) -> None:
    cfg = AppConfig(sim=SimConfig(n_players_list=[2], seed_list=[101]))
    cfg.screening.practical_delta_by_k = {2: 0.03}
    cfg.screening.delta_across_k = 0.03
    setattr(cfg.screening, field, value)

    with pytest.raises(ValueError, match=message):
        cfg.validate_statistical_contract()


@pytest.mark.parametrize(("batch_count", "minimum"), [(99, 30), (100, 29)])
def test_statistical_contract_locks_batch_construction(batch_count: int, minimum: int) -> None:
    cfg = AppConfig(sim=SimConfig(n_players_list=[2], seed_list=[101]))
    cfg.screening.practical_delta_by_k = {2: 0.03}
    cfg.screening.delta_across_k = 0.03
    cfg.batching.target_batches = batch_count
    cfg.batching.min_shuffles_per_batch = minimum

    with pytest.raises(ValueError, match="exactly 100 equal batches"):
        cfg.validate_statistical_contract()


def test_statistical_contract_requires_declared_practical_thresholds() -> None:
    cfg = AppConfig(sim=SimConfig(n_players_list=[2, 4], seed_list=[101, 202]))

    with pytest.raises(ValueError, match="practical_delta_by_k"):
        cfg.validate_statistical_contract(require_two_roots=True)


def test_statistical_contract_validates_declared_k_weights() -> None:
    cfg = AppConfig(sim=SimConfig(n_players_list=[2, 4], seed_list=[101, 202]))
    cfg.screening.practical_delta_by_k = {2: 0.03, 4: 0.015}
    cfg.screening.delta_across_k = 0.02
    cfg.k_aggregation.method = "declared-mapping"
    cfg.k_aggregation.k_weights = {2: 0.25, 4: 0.50}

    with pytest.raises(ValueError, match="sum to 1"):
        cfg.validate_statistical_contract(require_two_roots=True)


def test_freshness_key_covers_every_locked_statistical_dimension() -> None:
    cfg = AppConfig(sim=SimConfig(n_players_list=[4, 2]))

    freshness = cfg.freshness_key()

    assert freshness == {
        "artifact_contract_version": 1,
        "estimand_version": 1,
        "schema_version": 1,
        "rng_scheme_version": 1,
        "baseline_version": 1,
        "k_support_version": 1,
        "weighting_version": 1,
        "conditioning_version": 1,
        "multiplicity_version": 1,
        "candidate_family_version": 1,
        "baseline": "chance_rate_by_k",
        "required_player_counts": [2, 4],
        "k_aggregation_method": "equal-k",
        "k_weights": None,
        "conditioning": "unconditional_default",
        "multiplicity": "holm_h2h",
    }


@pytest.mark.parametrize(
    "version_field",
    [
        "artifact_contract_version",
        "estimand_version",
        "schema_version",
        "baseline_version",
        "k_support_version",
        "weighting_version",
        "conditioning_version",
        "multiplicity_version",
        "candidate_family_version",
    ],
)
def test_stage_freshness_hash_changes_for_each_contract_version(version_field: str) -> None:
    cfg = AppConfig(sim=SimConfig(n_players_list=[2, 4]))
    before = cfg.stage_config_sha("metrics")

    setattr(cfg.artifact_contract, version_field, 2)

    assert cfg.stage_config_sha("metrics") != before


def test_statistical_contract_rejects_nonpositive_freshness_version() -> None:
    cfg = AppConfig(sim=SimConfig(n_players_list=[2]))
    cfg.screening.practical_delta_by_k = {2: 0.03}
    cfg.screening.delta_across_k = 0.03
    cfg.artifact_contract.conditioning_version = 0

    with pytest.raises(ValueError, match="versions must all be positive"):
        cfg.validate_statistical_contract()


def test_stage_freshness_hash_changes_for_rng_support_and_declared_weights() -> None:
    cfg = AppConfig(sim=SimConfig(n_players_list=[2, 4]))
    baseline = cfg.stage_config_sha("combine")

    cfg.rng.scheme_version = 2
    assert cfg.stage_config_sha("combine") != baseline
    cfg.rng.scheme_version = 1

    cfg.sim.n_players_list = [2, 4, 12]
    assert cfg.stage_config_sha("combine") != baseline
    cfg.sim.n_players_list = [2, 4]

    cfg.k_aggregation.method = "declared-mapping"
    cfg.k_aggregation.k_weights = {2: 0.25, 4: 0.75}
    assert cfg.stage_config_sha("combine") != baseline


def test_load_rejects_retired_statistical_key(write_yaml) -> None:
    path = write_yaml("retired.yaml", {"head2head": {"fdr_q": 0.05}})

    with pytest.raises(ValueError, match="Retired config key 'head2head.fdr_q'"):
        load_app_config(path)


def test_load_rejects_retired_manual_shuffle_count(write_yaml) -> None:
    path = write_yaml("retired-shuffles.yaml", {"sim": {"num_shuffles": 100}})

    with pytest.raises(ValueError, match="screening.resolution_delta and batching settings"):
        load_app_config(path)


def test_load_builds_new_statistical_sections(write_yaml) -> None:
    path = write_yaml(
        "contract.yaml",
        {
            "sim": {"n_players_list": [2], "seed_list": [101]},
            "screening": {
                "practical_delta_by_k": {2: 0.03},
                "delta_across_k": 0.03,
                "max_shuffles_per_root_k": 10_000,
                "projected_games_per_second": 250.0,
            },
            "head2head": {"family_alpha": 0.02, "target_power": 0.8},
            "k_aggregation": {"method": "equal-k", "k_weights": None},
        },
    )

    cfg = load_app_config(path, seed_list_len=1)

    assert cfg.rng.bit_generator == "PCG64DXSM"
    assert cfg.screening.practical_delta_by_k == {2: 0.03}
    assert cfg.screening.max_shuffles_per_root_k == 10_000
    assert cfg.screening.projected_games_per_second == pytest.approx(250.0)
    assert cfg.head2head.family_alpha == pytest.approx(0.02)
    cfg.validate_statistical_contract()
