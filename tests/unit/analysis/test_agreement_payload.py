from __future__ import annotations

from pathlib import Path

import pandas as pd

from farkle.analysis import agreement
from farkle.analysis.agreement import MethodData


class _StubStageLog:
    def __init__(self) -> None:
        self.missing_calls: list[tuple[str, dict[str, object]]] = []

    def missing_input(self, reason: str, **extra: object) -> None:
        self.missing_calls.append((reason, extra))


class _GraphWithNodes:
    def number_of_nodes(self) -> int:
        return 0


def _mk_cfg(tmp_path: Path) -> agreement.AppConfig:
    cfg = agreement.AppConfig()
    cfg.io.results_dir_prefix = tmp_path / "results"
    return cfg


def test_build_payload_trueskill_errors_and_missing_inputs(tmp_path: Path, monkeypatch) -> None:
    cfg = _mk_cfg(tmp_path)
    stage_log = _StubStageLog()

    monkeypatch.setattr(agreement, "_load_trueskill", lambda *_args, **_kwargs: (_ for _ in ()).throw(FileNotFoundError("not found")))
    assert agreement._build_payload(cfg, players=2, pooled_scope=False, stage_log=stage_log) is None
    assert stage_log.missing_calls[-1][0] == "not found"

    monkeypatch.setattr(agreement, "_load_trueskill", lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad ts")))
    assert agreement._build_payload(cfg, players=2, pooled_scope=False, stage_log=stage_log) is None
    assert stage_log.missing_calls[-1][0] == "bad ts"

    monkeypatch.setattr(agreement, "_load_trueskill", lambda *_args, **_kwargs: None)
    assert agreement._build_payload(cfg, players=2, pooled_scope=False, stage_log=stage_log) is None
    assert stage_log.missing_calls[-1][0] == "missing TrueSkill ratings"


def test_build_payload_with_frequentist_error_and_strategy_filtering(tmp_path: Path, monkeypatch) -> None:
    cfg = _mk_cfg(tmp_path)
    cfg.analysis.agreement_strategies = ["s1", "s2"]
    stage_log = _StubStageLog()

    trueskill_data = MethodData(
        scores=pd.Series([3.0, 2.0, 1.0], index=["s1", "s2", "s3"]),
        tiers={"s1": 1, "s2": 2, "s3": 3},
        per_seed_scores=[
            pd.Series([3.1, 2.1, 1.1], index=["s1", "s2", "s3"]),
            pd.Series([2.9, 1.9, 0.9], index=["s1", "s2", "s3"]),
        ],
    )

    monkeypatch.setattr(agreement, "_load_trueskill", lambda *_args, **_kwargs: trueskill_data)
    monkeypatch.setattr(
        agreement,
        "_load_frequentist",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("freq bad")),
    )

    payload = agreement._build_payload(cfg, players=2, pooled_scope=False, stage_log=stage_log)

    assert payload is not None
    assert payload["methods"] == ["trueskill"]
    assert payload["strategy_counts"]["trueskill"] == 2
    assert payload["seed_stability"]["trueskill"]["strategies"] == 2
    assert payload["seed_stability"]["trueskill"]["seeds"] == 2
    assert stage_log.missing_calls[-1][0] == "freq bad"


def test_build_payload_pooled_h2h_error_and_missing(tmp_path: Path, monkeypatch) -> None:
    cfg = _mk_cfg(tmp_path)
    stage_log = _StubStageLog()
    trueskill_data = MethodData(
        scores=pd.Series([1.0, 2.0], index=["a", "b"]),
        tiers={"a": 1, "b": 2},
        per_seed_scores=[],
    )

    monkeypatch.setattr(agreement, "_load_trueskill", lambda *_args, **_kwargs: trueskill_data)
    monkeypatch.setattr(agreement, "_load_frequentist", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        agreement,
        "_load_head2head",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("h2h bad")),
    )
    payload = agreement._build_payload(cfg, players=0, pooled_scope=True, stage_log=stage_log)
    assert payload is not None
    assert payload["comparison_scope"]["h2h"] is None
    assert stage_log.missing_calls[-1][0] == "h2h bad"

    monkeypatch.setattr(agreement, "_load_head2head", lambda *_args, **_kwargs: None)
    payload = agreement._build_payload(cfg, players=0, pooled_scope=True, stage_log=stage_log)
    assert payload is not None
    assert payload["comparison_scope"]["h2h"] is None


def test_run_control_flow_variants(tmp_path: Path, monkeypatch) -> None:
    cfg = _mk_cfg(tmp_path)
    cfg.sim.n_players_list = [2]
    cfg.analysis.agreement_include_pooled = True

    missing_msgs: list[str] = []

    class _RunStageLog:
        def start(self) -> None:
            return None

        def missing_input(self, reason: str, **_extra: object) -> None:
            missing_msgs.append(reason)

    monkeypatch.setattr(agreement, "stage_logger", lambda *_args, **_kwargs: _RunStageLog())

    monkeypatch.setattr(agreement, "_build_payload", lambda *_args, **_kwargs: None)
    agreement.run(cfg)
    assert missing_msgs[-1] == "no agreement payloads generated"
    assert not (cfg.agreement_stage_dir / "agreement_summary.parquet").exists()

    def _per_k_only(_cfg: agreement.AppConfig, players: int, pooled_scope: bool, **_kwargs: object):
        if pooled_scope:
            return None
        return {
            "methods": ["trueskill"],
            "comparison_scope": {"mode": "per_k"},
            "strategy_counts": {"trueskill": 2},
            "coverage": None,
            "spearman": None,
            "kendall": None,
            "ari": None,
            "nmi": None,
            "seed_stability": {"trueskill": None},
        }

    monkeypatch.setattr(agreement, "_build_payload", _per_k_only)
    agreement.run(cfg)
    summary = pd.read_parquet(cfg.agreement_stage_dir / "agreement_summary.parquet")
    assert len(summary) == 1
    assert summary.iloc[0]["players"] == 2
    assert cfg.agreement_output_path(2).exists()

    def _pooled_only(_cfg: agreement.AppConfig, players: int, pooled_scope: bool, **_kwargs: object):
        if not pooled_scope:
            return None
        return {
            "methods": ["trueskill", "h2h"],
            "comparison_scope": {"mode": "pooled", "h2h": "pooled"},
            "strategy_counts": {"trueskill": 3, "h2h": 3},
            "coverage": {"h2h_vs_trueskill": {"common": 3}},
            "spearman": {"h2h_vs_trueskill": 1.0},
            "kendall": {"h2h_vs_trueskill": 1.0},
            "ari": None,
            "nmi": None,
            "seed_stability": {"trueskill": None, "h2h": None},
        }

    monkeypatch.setattr(agreement, "_build_payload", _pooled_only)
    agreement.run(cfg)
    summary = pd.read_parquet(cfg.agreement_stage_dir / "agreement_summary.parquet")
    assert len(summary) == 1
    assert summary.iloc[0]["players"] == "pooled"
    assert cfg.agreement_output_path_pooled().exists()

    cfg.sim.n_players_list = [2, 3]
    cfg.analysis.agreement_include_pooled = False

    def _both(_cfg: agreement.AppConfig, players: int, pooled_scope: bool, **_kwargs: object):
        if pooled_scope:
            return None
        return {
            "methods": ["trueskill"],
            "comparison_scope": {"mode": "per_k"},
            "strategy_counts": {"trueskill": players},
            "coverage": {"x": {"common": players}},
            "spearman": {"x": float(players)},
            "kendall": {"x": float(players)},
            "ari": {"x": players / 10.0},
            "nmi": {"x": players / 20.0},
            "seed_stability": {
                "trueskill": {"top_strategies": [{"strategy": f"s{players}", "stddev": 1.0}]}
            },
        }

    monkeypatch.setattr(agreement, "_build_payload", _both)
    agreement.run(cfg)
    summary = pd.read_parquet(cfg.agreement_stage_dir / "agreement_summary.parquet")
    assert len(summary) == 2
    assert set(summary["players"]) == {2, 3}
    three_row = summary.loc[summary["players"] == 3].iloc[0]
    assert three_row["spearman_x"] == 3.0
    assert (
        three_row["seed_stability_trueskill_top_strategies"]
        == '[{"stddev": 1.0, "strategy": "s3"}]'
    )


def test_trueskill_resolver_edge_cases(tmp_path: Path) -> None:
    cfg = _mk_cfg(tmp_path)

    assert agreement._resolve_trueskill_per_k_path(cfg, "pooled") is None

    missing_path = agreement._resolve_trueskill_per_k_path(cfg, 4)
    assert missing_path is not None
    assert missing_path.name == "ratings_4.parquet"

    pooled_dir = cfg.trueskill_stage_dir / "pooled"
    pooled_dir.mkdir(parents=True, exist_ok=True)
    stage_dir = cfg.trueskill_stage_dir
    stage_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = cfg.analysis_dir
    analysis_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"strategy": ["a"], "mu": [1.0]}).to_parquet(
        analysis_dir / "ratings_k_weighted_seed7.parquet"
    )
    pd.DataFrame({"strategy": ["a"], "mu": [2.0]}).to_parquet(
        stage_dir / "ratings_k_weighted_seed7.parquet"
    )
    pd.DataFrame({"strategy": ["a"], "mu": [3.0]}).to_parquet(
        pooled_dir / "ratings_k_weighted_seed7.parquet"
    )
    (pooled_dir / "ratings_k_weighted_seedBAD.parquet").write_text("ignore")

    pooled_paths = agreement._resolve_trueskill_seed_paths(cfg, players=0, pooled_scope=True)
    assert len(pooled_paths) == 1
    assert pooled_paths[0].parent.name == "pooled"

    assert agreement._resolve_trueskill_seed_paths(cfg, players="pooled", pooled_scope=False) == []


def test_tier_and_rank_and_seed_helpers_degenerate_cases() -> None:
    ari, nmi = agreement._tier_agreements({})
    assert ari is None
    assert nmi is None

    ari, nmi = agreement._tier_agreements({"a": None, "b": {"s1": 1, "s2": 2}})
    assert ari == {"a_vs_b": None}
    assert nmi == {"a_vs_b": None}

    ari, nmi = agreement._tier_agreements({"a": {"s1": 1}, "b": {"s1": 1}})
    assert ari == {"a_vs_b": None}
    assert nmi == {"a_vs_b": None}

    ari, nmi = agreement._tier_agreements({"a": {"s1": 1, "s2": 1}, "b": {"s1": 2, "s2": 2}})
    assert ari == {"a_vs_b": None}
    assert nmi == {"a_vs_b": None}

    spearman, kendall, coverage = agreement._rank_correlations({})
    assert spearman is None
    assert kendall is None
    assert coverage == {}

    constant = {
        "a": pd.Series([1.0, 1.0], index=["s1", "s2"]),
        "b": pd.Series([2.0, 2.0], index=["s1", "s2"]),
    }
    spearman, kendall, _ = agreement._rank_correlations(constant)
    assert spearman == {"a_vs_b": None}
    assert kendall == {"a_vs_b": None}

    assert agreement._summarize_seed_stability([]) is None
    assert (
        agreement._summarize_seed_stability(
            [pd.Series([1.0], index=["x"]), pd.Series([2.0], index=["y"])]
        )
        is None
    )

    summaries = agreement._summarize_seed_stability(
        [
            pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], index=["a", "b", "c", "d", "e", "f"]),
            pd.Series([2.0, 4.0, 4.0, 7.0, 5.0, 8.0], index=["a", "b", "c", "d", "e", "f"]),
        ]
    )
    assert summaries is not None
    top = summaries["top_strategies"]
    assert len(top) == 5
    assert top[0]["stddev"] >= top[-1]["stddev"]


def test_load_head2head_null_path_variants(tmp_path: Path, monkeypatch) -> None:
    cfg = _mk_cfg(tmp_path)

    assert agreement._load_head2head(cfg) is None

    decisions_path = cfg.post_h2h_path("bonferroni_decisions.parquet")
    decisions_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"a": "a", "b": "b", "is_sig": True}]).to_parquet(decisions_path)

    monkeypatch.setattr(agreement, "build_significant_graph", lambda _df: _GraphWithNodes())
    assert agreement._load_head2head(cfg) is None

    class _GraphSomeNodes:
        def number_of_nodes(self) -> int:
            return 2

    monkeypatch.setattr(agreement, "build_significant_graph", lambda _df: _GraphSomeNodes())
    monkeypatch.setattr(agreement, "derive_sig_ranking", lambda _graph: [])
    assert agreement._load_head2head(cfg) is None
