"""Plan-specific operational H2H consent, outside statistical configuration."""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Iterator, Mapping
from contextvars import ContextVar
from pathlib import Path
from typing import Any, cast

from farkle.config import AppConfig, ArtifactScope
from farkle.utils.artifact_contract import ArtifactContractError, make_artifact_sidecar, sha256_file
from farkle.utils.artifacts import read_json_artifact, write_json_artifact_atomic
from farkle.utils.authenticated_contract import (
    AuthenticatedContractError,
    load_authenticated_sidecar,
)

_POLICY: ContextVar[tuple[bool, str | None]] = ContextVar("h2h_cap_policy", default=(False, None))
PROMPT = "Override the cap for this H2H plan and continue? [Y/N]"


class H2HAuthorizationRequired(RuntimeError):
    """A resumable operational stop, not a simulation or statistical failure."""

    def __init__(self, reason: str, instruction: str) -> None:
        self.reason = reason
        self.instruction = instruction
        super().__init__(f"H2H blocked_by_cap ({reason}). {instruction}")


@contextlib.contextmanager
def cap_authorization_policy(
    *, interactive: bool = False, override: str | None = None
) -> Iterator[None]:
    """Scope CLI consent to this invocation; never serialize it as public config."""
    token = _POLICY.set((interactive, override))
    try:
        yield
    finally:
        _POLICY.reset(token)


def authorization_token(plan: Mapping[str, Any]) -> str:
    return f"{plan['schedule_hash']}:{plan['maximum_total_attempts']}"


def budget_description(plan: Mapping[str, Any], cap: int | None) -> str:
    maximum = int(plan["maximum_total_attempts"])
    percentages = (
        "required budget: unlimited cap; percentage over cap: 0%"
        if cap is None
        else (
            f"required budget: {100 * maximum / cap:.2f}% of cap; "
            f"percentage over cap: {100 * max(0, maximum - cap) / cap:.2f}%"
            if cap > 0
            else "required budget: undefined for zero cap; percentage over cap: undefined"
        )
    )
    return (
        f"H2H plan {plan['schedule_hash']}\n"
        f"Target completed games: {int(plan['total_completed_required']):,}\n"
        f"Maximum attempts including safety replacements: {maximum:,}\n"
        f"Configured cap: {cap if cap is not None else 'unlimited'}; {percentages}"
    )


def validate_authorization(
    authorization: Mapping[str, Any], plan: Mapping[str, Any], *, require_ready: bool = False
) -> None:
    """Validate amount and exact frozen identity after authenticating the bytes."""
    if authorization.get("schema_version") != 1:
        raise ValueError("unsupported H2H cap authorization schema")
    for field in ("family_hash", "schedule_hash", "maximum_total_attempts", "total_block_count"):
        if authorization.get(field) != plan.get(field):
            raise ValueError(f"H2H cap authorization conflicts with immutable plan field {field}")
    source = authorization.get("authorization_source")
    cap = authorization.get("configured_total_game_cap")
    amount = authorization.get("authorized_total_game_cap")
    maximum = int(plan["maximum_total_attempts"])
    if amount is not None and (type(amount) is not int or amount < 0):
        raise ValueError("invalid authorized H2H attempt amount")
    if cap is not None and (type(cap) is not int or cap < 0):
        raise ValueError("invalid configured H2H cap")
    if source in {"interactive", "explicit_override"}:
        if type(amount) is not int or amount != maximum:
            raise ValueError("H2H override must authorize exactly the maximum attempts")
        ready = True
    elif source == "configured_cap":
        if amount != cap:
            raise ValueError("configured H2H authorization amount differs from cap")
        ready = cap is None or maximum <= cap
    else:
        raise ValueError("invalid H2H authorization source")
    if authorization.get("execution_authorization") != ("ready" if ready else "blocked_by_cap"):
        raise ValueError("H2H authorization status disagrees with authorized amount")
    if require_ready and not ready:
        raise ValueError("H2H execution lacks sufficient authenticated authorization")


def validate_authorization_sources(
    authorization: Mapping[str, Any], power_path: Path, manifest_path: Path
) -> None:
    """Bind consent to exact immutable files as well as their semantic hashes."""
    for field, path in (
        ("power_plan_sha256", power_path),
        ("block_manifest_sha256", manifest_path),
    ):
        if authorization.get(field) != sha256_file(path):
            raise ValueError(f"H2H authorization source changed: {field}")


def _publish(cfg: AppConfig, plan: Mapping[str, Any], payload: dict[str, Any]) -> None:
    from farkle.analysis.h2h_schedule import h2h_method_contract

    path = cfg.h2h_cap_authorization_path()
    sidecar = make_artifact_sidecar(
        cfg,
        path,
        producer="h2h_schedule",
        scope=ArtifactScope.H2H_2P,
        source_scope=ArtifactScope.H2H_2P,
        operation="h2h_cap_authorization",
        consistency_columns=list(payload),
        grouping_keys=["family_hash", "schedule_hash"],
        source_artifacts=[cfg.h2h_power_plan_path(), cfg.h2h_block_manifest_path()],
        player_counts=[2],
        required_player_counts=[2],
        missing_cell_policy="fail",
        seed_scope="both_roots_combined" if len(plan["root_seeds"]) == 2 else "single_root",
        method_contract=h2h_method_contract(
            cfg,
            "h2h_cap_authorization",
            family_hash=str(plan["family_hash"]),
            schedule_hash=str(plan["schedule_hash"]),
        ),
    )
    write_json_artifact_atomic(payload, path, sidecar=sidecar)


def resolve_cap_authorization(
    cfg: AppConfig, plan: Mapping[str, Any], *, request: bool = False
) -> dict[str, Any]:
    """Reuse authenticated consent or obtain it before any more games execute.

    Interrupted data/sidecar publication is never approval. Reconstruct only
    configured-cap permission, then require new explicit consent when exceeded.
    Valid records for a different plan fail closed instead of being reassigned.
    """
    path = cfg.h2h_cap_authorization_path()
    cap = cfg.head2head.total_game_cap
    existing: dict[str, Any] | None = None
    if path.exists():
        try:
            existing = cast(
                dict[str, Any],
                read_json_artifact(
                    path,
                    expected_sidecar={
                        "scope": ArtifactScope.H2H_2P.value,
                        "operation": "h2h_cap_authorization",
                    },
                ),
            )
        except (ArtifactContractError, AuthenticatedContractError):
            existing = None
        if existing is not None:
            if cfg.artifact_contract.artifact_contract_version == 3:
                metadata = load_authenticated_sidecar(path)
                if (
                    cfg._code_identity is not None
                    and metadata.stage_identity.code != cfg._code_identity
                ):
                    raise ValueError("H2H authorization belongs to different code identity")
                if (
                    cfg._run_lineage_sha256 is not None
                    and metadata.stage_identity.immutable_design_identities.get(
                        "run_lineage_sha256"
                    )
                    != cfg._run_lineage_sha256
                ):
                    raise ValueError("H2H authorization belongs to a different run lineage")
            validate_authorization(existing, plan)
            validate_authorization_sources(
                existing, cfg.h2h_power_plan_path(), cfg.h2h_block_manifest_path()
            )
            if existing["configured_total_game_cap"] != cap:
                existing = None
    if existing is None:
        maximum = int(plan["maximum_total_attempts"])
        existing = {
            "schema_version": 1,
            "power_plan_sha256": sha256_file(cfg.h2h_power_plan_path()),
            "block_manifest_sha256": sha256_file(cfg.h2h_block_manifest_path()),
            **{
                key: plan[key]
                for key in (
                    "family_hash",
                    "schedule_hash",
                    "maximum_total_attempts",
                    "total_block_count",
                )
            },
            "configured_total_game_cap": cap,
            "authorized_total_game_cap": cap,
            "authorization_source": "configured_cap",
            "execution_authorization": (
                "ready" if cap is None or maximum <= cap else "blocked_by_cap"
            ),
        }
        _publish(cfg, plan, existing)
    if request:
        print(budget_description(plan, cap), flush=True)
    if not request or existing["execution_authorization"] == "ready":
        return existing
    instruction = (
        "Completed work is preserved. Resume the same command in a terminal and answer Y, "
        f"or add the global option --h2h-authorize {authorization_token(plan)} "
        "before the command name. The configured cap and statistical plan stay unchanged."
    )
    interactive, override = _POLICY.get()
    if override is not None:
        if override != authorization_token(plan):
            raise H2HAuthorizationRequired("override_mismatch", instruction)
        source = "explicit_override"
    elif interactive and sys.stdin is not None and sys.stdin.isatty():
        while True:
            try:
                answer = input(PROMPT + " ").strip().lower()
            except EOFError:
                raise H2HAuthorizationRequired("eof", instruction) from None
            except KeyboardInterrupt:
                raise H2HAuthorizationRequired("interrupted", instruction) from None
            if answer == "y":
                break
            if answer in {"n", ""}:
                raise H2HAuthorizationRequired("refused" if answer else "blank", instruction)
            print("Please enter Y or N. Blank input declines.", flush=True)
        source = "interactive"
    else:
        raise H2HAuthorizationRequired("noninteractive", instruction)
    approved = {
        **existing,
        "authorized_total_game_cap": int(plan["maximum_total_attempts"]),
        "authorization_source": source,
        "execution_authorization": "ready",
    }
    _publish(cfg, plan, approved)
    # Authenticate the completed publication before handing control to execution.
    persisted = cast(
        dict[str, Any],
        read_json_artifact(
            path,
            expected_sidecar={
                "scope": ArtifactScope.H2H_2P.value,
                "operation": "h2h_cap_authorization",
            },
        ),
    )
    validate_authorization(persisted, plan, require_ready=True)
    return persisted
