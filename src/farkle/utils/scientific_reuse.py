"""Compare authenticated producer evidence against the current invocation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

from farkle.config import AppConfig
from farkle.utils.authenticated_contract import (
    ArtifactMismatchError,
    CodeIdentity,
    StageIdentity,
)
from farkle.utils.scientific_dependencies import stage_producer
from farkle.utils.scientific_invocation import current, dependency_key


def closure(stage: StageIdentity) -> dict[str, str]:
    return {k: v for k, v in stage.immutable_design_identities.items() if k.startswith("reuse_v1/")}


def compatible_producer(stage: StageIdentity, code: CodeIdentity) -> bool:
    """Older sidecars retain the exact Git guard. Only a live pin admits another revision."""
    if stage.code != code:
        return launch_matches(stage)
    pin = current()
    # Older same-revision publications keep their original freshness contract.
    return pin is None or not closure(stage) or launch_matches(stage)


def launch_matches(stage: StageIdentity) -> bool:
    pin = current()
    roots = stage.stage_config.selected_config.get("sim.seed_list")
    if pin is None or not isinstance(roots, (tuple, list)) or not roots:
        return False
    evidence = closure(stage)
    suffix = f"/{stage.stage_key}/" + ",".join(map(str, roots))
    own = stage.immutable_design_identities.get("stage_dependencies_v1")
    return (
        any(k.endswith(suffix) and v == own for k, v in evidence.items())
        and evidence.items() <= pin.launch_digests.items()
    )


def publication_closure(
    cfg: AppConfig,
    stage: str,
    dependencies: str,
    sources: Iterable[StageIdentity],
    *,
    classified_leaf: bool,
) -> Mapping[str, str]:
    pin = current()
    if pin is None:
        return {}
    key = dependency_key(stage_producer(cfg, stage), cfg.sim.seed_list or [cfg.sim.seed])
    if pin.launch_digests.get(key) != dependencies:
        return {}
    result = {key: dependencies}
    for source in sources:
        inherited = closure(source)
        if not launch_matches(source):
            return {}
        result.update(inherited)
        classified_leaf = True
    return result if classified_leaf else {}


# Root publications read by the pair contributions, plus their direct and
# transitive producer inputs. Rolling aggregate checkpoints remain resumable.
ROOT_INPUT_OPERATIONS = frozenset(
    {
        "publish_strategy_manifest",
        "publish_simulation_workload_plan",
        "publish_simulation_row_shard",
        "publish_simulation_shard_manifest",
        "ingest_simulation_rows",
        "ingest_simulation_rows_stream_manifest",
        "curate_game_rows",
        "aggregate_player_batch_statistics",
        "materialize_performance_batch_matrix",
        "materialize_root_stability_batch_matrix",
        "sequential_rating",
    }
)


def sealed_root_input(path: Path) -> bool:
    from farkle.utils.artifact_contract import read_json_file_with_retry, sidecar_path

    adjacent = sidecar_path(path)
    if not adjacent.exists():
        return False
    raw = read_json_file_with_retry(adjacent)
    return raw.get("artifact_contract_version") == 3 and (
        raw.get("method_contract", {}).get("procedure") in ROOT_INPUT_OPERATIONS
    )


def immutable_input_error(path: Path) -> ArtifactMismatchError:
    return ArtifactMismatchError(
        f"Immutable scientific input differs at {path}; use a new output root"
    )
