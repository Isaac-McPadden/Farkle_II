import warnings
from dataclasses import asdict

from farkle.game.engine import GameMetrics, GameStats, PlayerStats


def test_players_dict_and_per_player_alias():
    players = {
        "A": PlayerStats(
            score=10,
            farkles=1,
            rolls=2,
            highest_turn=5,
            strategy="x",
            rank=1,
            loss_margin=0,
        )
    }
    game = GameStats(
        n_players=1,
        table_seed=42,
        n_rounds=1,
        total_rolls=2,
        total_farkles=1,
        margin=0,
    )
    gm = GameMetrics(players, game)

    expected = {"A": asdict(players["A"])}
    assert gm.players_dict == expected

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert gm.per_player == expected
        assert any(issubclass(rec.category, DeprecationWarning) for rec in w)
