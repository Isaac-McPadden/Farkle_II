"""Phase 4A is a read-only projection. No historical artifacts are needed here."""

from dataclasses import dataclass, replace
from pathlib import Path
from shutil import copyfile

import pytest

from farkle.analysis.stage_registry import (
    resolve_root_pair_stage_layout,
    resolve_scoped_stage_definition,
    resolve_stage_definition,
    resolve_stage_layout,
)
from farkle.config import AppConfig, SimConfig
from farkle.utils import scientific_identity
from farkle.utils.authenticated_contract import identity_sha256
from farkle.utils.scientific_dependencies import Producer, dependencies_for
from farkle.utils.scientific_identity import (
    Compatibility,
    NumericalEnvironment,
    ScientificFingerprint,
    ScientificInput,
    project_scientific_identity,
    scientific_compatibility,
)

ROOT = Path(__file__).resolve().parents[3]
SIM = Producer("root", "simulation")
RATING = Producer("root", "trueskill")
PAIR_RATING = Producer("pair", "trueskill")
RNG = Producer("root", "rng_diagnostics")


def evidence(producer, cfg=None):
    deps = dependencies_for(producer)
    env = NumericalEnvironment(
        "a" * 64, {name: identity_sha256(name) for name in deps.distributions}
    )
    upstream = ScientificFingerprint(1, "b" * 64)
    sources = {
        role: ScientificInput(identity_sha256(role), upstream)
        for role in deps.required_inputs(cfg or AppConfig())
    }
    return env, sources


def project(producer=SIM, cfg=None, repo=ROOT, **overrides):
    cfg = cfg or AppConfig()
    if producer.scope == "pair" and cfg.sim.seed_list is None:
        cfg.sim.seed_list = [cfg.sim.seed, 1]
    environment, inputs = evidence(producer, cfg)
    kwargs = {"environment": environment, "inputs": inputs, **overrides}
    return project_scientific_identity(producer, cfg, repo_root=repo, **kwargs)


def digest(*args, **kwargs):
    result = project(*args, **kwargs)
    assert result.sha256 is not None, result.unresolved
    return result.sha256


def test_scoped_registry_and_complete_catalog():
    cfg = AppConfig()
    for placement in resolve_stage_layout(cfg).placements:
        assert dependencies_for(Producer("root", placement.definition.key)).input_roles
    for placement in resolve_root_pair_stage_layout(cfg).placements:
        assert dependencies_for(Producer("pair", placement.definition.key)).input_roles
    assert resolve_scoped_stage_definition("root", "trueskill").depends_on == ("curate",)
    assert resolve_stage_definition("trueskill") == resolve_scoped_stage_definition(
        "pair", "trueskill"
    )
    assert "ordered_curated_rows" in dependencies_for(RATING).input_roles
    assert dependencies_for(PAIR_RATING).input_roles == {"root_rating_cells"}
    assert digest(RATING) != digest(PAIR_RATING)
    with pytest.raises(KeyError):
        dependencies_for(Producer("pair", "simulation"))


@pytest.fixture
def source_tree(tmp_path):
    for file in dependencies_for(SIM).source_files | dependencies_for(RATING).source_files:
        dest = tmp_path / file
        dest.parent.mkdir(parents=True, exist_ok=True)
        copyfile(ROOT / file, dest)
    return tmp_path


def test_source_and_documentation_changes(source_tree):
    original_sim = digest(repo=source_tree)
    original_rating = digest(RATING, repo=source_tree)
    (source_tree / "README.md").write_text("Revised operator instructions.")
    assert digest(repo=source_tree) == original_sim
    assert digest(RATING, repo=source_tree) == original_rating

    (source_tree / "src/farkle/analysis/run_trueskill.py").write_text(
        "# changed rating implementation"
    )
    assert digest(repo=source_tree) == original_sim
    assert digest(RATING, repo=source_tree) != original_rating


@pytest.mark.parametrize(
    "file",
    [
        "game/engine.py",
        "game/scoring.py",
        "simulation/strategies.py",
        "utils/random.py",
        "simulation/run_tournament.py",
        "utils/parallel.py",
        "utils/aggregation.py",
    ],
)
def test_worker_engine_strategy_rng_and_shared_source(source_tree, file):
    original = digest(repo=source_tree)
    (source_tree / "src/farkle" / file).write_text("# changed implementation")
    assert digest(repo=source_tree) != original


def test_execution_provenance_and_release_controls():
    cfg = AppConfig()
    producers = (
        SIM,
        Producer("root", "ingest"),
        Producer("root", "h2h_power"),
        Producer("root", "h2h_execute"),
        Producer("root", "reporting"),
    )
    original = [digest(p, cfg) for p in producers]
    cfg.io.results_dir_prefix = Path("relocated")
    cfg.io.temp_dir = Path("scratch")
    cfg.sim.n_jobs = 4
    cfg.sim.ckpt_every_sec += 1
    cfg.ingest.n_jobs += 1
    cfg.ingest.parquet_codec = "gzip"
    cfg.resources.aggregate_memory_hard_limit_mb += 512
    cfg.head2head.n_jobs += 1
    cfg.head2head.total_game_cap = 30
    cfg.profile.purpose = "development"
    cfg.config_sha = "1" * 64
    cfg._run_lineage_sha256 = "2" * 64
    assert [digest(p, cfg) for p in producers] == original


def test_topology_and_effective_batch_bound():
    cfg = AppConfig()
    cfg.sim.n_jobs = 2
    original_rng = digest(RNG, cfg)
    cfg.sim.n_jobs = 4
    assert digest(RNG, cfg) != original_rng
    stats = Producer("root", "game_stats")
    original = digest(stats, cfg)
    cfg.resources.stage_batch_bytes["game_stats"] //= 2
    assert digest(stats, cfg) != original


def test_pair_requires_every_root_rating_cell():
    cfg = AppConfig()
    cfg.sim.seed_list = [0, 1]
    cfg.sim.n_players_list = [2, 5]
    _, sources = evidence(PAIR_RATING, cfg)
    assert set(sources) == {f"root_rating_cells/{root}/{k}" for root in (0, 1) for k in (2, 5)}
    complete = project(PAIR_RATING, cfg, inputs=sources)
    sources.pop("root_rating_cells/1/2")
    partial = project(PAIR_RATING, cfg, inputs=sources)
    assert complete.classified
    assert scientific_compatibility(complete, partial) is Compatibility.UNCLASSIFIED


def test_observed_distribution_version_is_scoped(monkeypatch):
    original_version = scientific_identity.version
    rating, sim = digest(RATING), digest(SIM)
    monkeypatch.setattr(
        scientific_identity,
        "version",
        lambda name: "other" if name == "trueskill" else original_version(name),
    )
    assert digest(RATING) != rating
    assert digest(SIM) == sim


def test_scientific_configuration_is_scoped():
    cfg = AppConfig()
    sim, rating = digest(SIM, cfg), digest(RATING, cfg)
    cfg.trueskill.beta += 1
    assert digest(SIM, cfg) == sim
    assert digest(RATING, cfg) != rating
    cfg.screening.resolution_delta /= 2
    assert digest(SIM, cfg) != sim
    cfg.sim.row_dir = Path("shards")
    enabled = digest(SIM, cfg)
    cfg.sim.row_dir = Path("other")
    assert digest(SIM, cfg) == enabled
    cfg.sim.row_dir = None
    assert digest(SIM, cfg) != enabled
    original = digest(SIM, cfg)
    cfg._game_profile_sha256 = "c" * 64
    assert digest(SIM, cfg) != original


def test_frozen_family_membership_alone_is_insufficient():
    producer = Producer("pair", "h2h_power")
    _, sources = evidence(producer)
    original = project(producer, inputs=sources)
    sources["frozen_membership"] = replace(
        sources["frozen_membership"], producer_science=ScientificFingerprint(1, "f" * 64)
    )
    assert (
        scientific_compatibility(original, project(producer, inputs=sources))
        is Compatibility.CHANGED
    )
    for missing in (None, ScientificFingerprint(99, "f" * 64)):
        sources["frozen_membership"] = replace(
            sources["frozen_membership"], producer_science=missing
        )
        assert (
            scientific_compatibility(original, project(producer, inputs=sources))
            is Compatibility.UNCLASSIFIED
        )
    del sources["frozen_family_manifest"]
    assert (
        scientific_compatibility(original, project(producer, inputs=sources))
        is Compatibility.UNCLASSIFIED
    )


def test_simulation_can_start_from_resolved_local_design():
    _, sources = evidence(SIM)
    local = {role: ScientificInput(value.semantic_sha256) for role, value in sources.items()}
    first = digest(inputs=local)
    local["resolved_strategy_roster"] = ScientificInput(identity_sha256(["different strategy"]))
    assert digest(inputs=local) != first


def test_native_and_distribution_builds():
    env, _ = evidence(RATING)
    original = project(RATING, environment=env)
    for new in (
        replace(env, runtime_sha256="c" * 64),
        replace(env, distribution_sha256={**env.distribution_sha256, "trueskill": "d" * 64}),
    ):
        assert (
            scientific_compatibility(original, project(RATING, environment=new))
            is Compatibility.CHANGED
        )
    for incomplete in (
        None,
        replace(env, distribution_sha256={}),
        replace(env, distribution_sha256={**env.distribution_sha256, "numpy": "1.26"}),
    ):
        assert (
            scientific_compatibility(original, project(RATING, environment=incomplete))
            is Compatibility.UNCLASSIFIED
        )


def test_partition_is_qualified_by_owner_family_and_complete_coordinates():
    producer = Producer("root", "simulation", "shuffle")
    first = digest(producer, coordinates=(5, 0))
    assert digest(producer, coordinates=(5, 1)) != first
    assert digest(SIM) != first
    for bad in ((), (5,), (6, 0), (5, True)):
        assert project(producer, coordinates=bad).sha256 is None
    assert project(SIM, coordinates=(5, 0)).sha256 is None
    with pytest.raises(KeyError):
        dependencies_for(Producer("pair", "trueskill", "shuffle"))
    tiled = Producer("root", "metrics", "outcome_precision")
    assert digest(tiled, coordinates=(5, "vectors", 0)) != digest(tiled, coordinates=(5, ".", 0))
    assert project(tiled, coordinates=(5, "new_pass", 0)).sha256 is None


def test_canonical_equivalent_configs_and_input_order():
    cfg = AppConfig()
    cfg.k_aggregation.k_weights = {2: 0.5, 5: 0.5}
    producer = Producer("root", "metrics")
    env, inputs = evidence(producer)
    first = project(producer, cfg, environment=env, inputs=inputs)
    cfg.k_aggregation.k_weights = {5: 0.5, 2: 0.5}
    cfg.sim.seed_list = [cfg.sim.seed]
    second = project(
        producer,
        cfg,
        environment=replace(
            env, distribution_sha256=dict(reversed(list(env.distribution_sha256.items())))
        ),
        inputs=dict(reversed(inputs.items())),
    )
    assert scientific_compatibility(first, second) is Compatibility.COMPATIBLE


def test_unknown_package_file_or_missing_required_source_blocks(source_tree):
    module = source_tree / "src/farkle/new_worker.pyd"
    module.write_bytes(b"new native code")
    unknown = project(repo=source_tree)
    assert unknown.unresolved and module.name in unknown.unresolved
    assert scientific_compatibility(unknown, unknown) is Compatibility.UNCLASSIFIED
    module.unlink()
    (source_tree / "src/farkle/game/engine.py").unlink()
    assert project(repo=source_tree).sha256 is None


def test_unclassified_config_or_version_blocks():
    @dataclass
    class ExtendedSim(SimConfig):
        new_method: str = "experimental"

    cfg = AppConfig(sim=ExtendedSim())
    result = project(cfg=cfg)
    assert result.unresolved and "sim.new_method" in result.unresolved
    cfg = AppConfig()
    cfg.sim.per_n = {5: SimConfig()}
    assert project(cfg=cfg).sha256 is None
    cfg.sim.per_n.clear()
    cfg.analysis.outputs["new_option"] = "unknown"
    assert project(cfg=cfg).sha256 is None
    current = project()
    for invalid in (
        ScientificFingerprint(99, current.sha256),
        ScientificFingerprint(1, ""),
        ScientificFingerprint(1, None),
    ):
        assert scientific_compatibility(invalid, invalid) is Compatibility.UNCLASSIFIED
