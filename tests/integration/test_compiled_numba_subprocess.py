from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path


def test_clean_subprocess_uses_normal_numba_for_scoring_and_seeded_simulation(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    source_root = repo_root / "src"
    script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(source_root)!r})

        from numba.core.registry import CPUDispatcher
        from farkle.game.scoring import _faces_to_counts_nb, score_roll_cached
        from farkle.simulation.simulation import _play_game
        from farkle.simulation.strategies import ThresholdStrategy
        from farkle.utils.random import RNG_SCHEME_VERSION, RandomPurpose

        counts = _faces_to_counts_nb.__call__(
            __import__("numpy").asarray([1, 1, 1, 5, 2, 3], dtype="int64")
        )
        assert isinstance(_faces_to_counts_nb, CPUDispatcher)
        assert _faces_to_counts_nb.signatures
        assert counts == (3, 1, 1, 0, 1, 0)
        score, used, _, single_fives, single_ones = score_roll_cached([1, 1, 1, 5, 2, 3])
        assert (score, used, single_fives, single_ones) == (350, 4, 1, 0)

        strategies = [
            ThresholdStrategy(score_threshold=0, dice_threshold=6, strategy_id=1),
            ThresholdStrategy(score_threshold=0, dice_threshold=6, strategy_id=2),
        ]
        provenance = {{
            "root_seed": 91,
            "k": 2,
            "shuffle_index": 3,
            "game_index": 0,
            "deterministic_batch_id": 1,
            "shuffle_seed": 17,
            "game_seed": 23,
            "rng_scheme_version": RNG_SCHEME_VERSION,
            "rng_purpose_namespace": int(RandomPurpose.TOURNAMENT_GAME),
        }}
        first = dict(_play_game(23, strategies, target_score=200, provenance=provenance))
        second = dict(_play_game(23, strategies, target_score=200, provenance=provenance))
        assert first == second
        assert first["root_seed"] == 91
        assert first["k"] == 2
        assert first["game_index"] == 0
        assert {{first["P1_strategy"], first["P2_strategy"]}} == {{1, 2}}
        print("compiled-ok")
        """
    )
    environment = os.environ.copy()
    environment["NUMBA_DISABLE_JIT"] = "0"
    environment["NUMBA_CACHE_DIR"] = str(tmp_path / "numba-cache")

    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "compiled-ok"
