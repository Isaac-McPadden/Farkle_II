import json
import os
import pickle
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from farkle import run_rf, run_trueskill


def test_analysis_pipeline(tmp_path):
    data_root = tmp_path / "data"
    res_dir = data_root / "results/2_players"
    res_dir.mkdir(parents=True)

    keepers = np.array(["A", "B", "C"])
    np.save(res_dir / "keepers_2.npy", keepers)

    rng = np.random.default_rng(0)
    winners = rng.choice(keepers, size=50)
    pd.DataFrame({"winner_strategy": winners}).to_csv(res_dir / "winners.csv", index=False)

    metrics = pd.DataFrame({
        "strategy": keepers,
        "feat1": [1, 2, 3],
        "feat2": [4, 5, 6],
    })
    metrics.to_parquet(data_root / "metrics.parquet")

    (data_root / "results/manifest.yaml").write_text(yaml.safe_dump({"seed": 0}))

    nb_dir = tmp_path / "notebooks"
    nb_dir.mkdir()
    src_nb = Path(__file__).resolve().parents[1] / "notebooks" / "farkle_report.ipynb"
    shutil.copy(src_nb, nb_dir / "farkle_report.ipynb")
    (tmp_path / "src").symlink_to(Path(__file__).resolve().parents[1] / "src")

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        run_trueskill.main()
        run_rf.main()

        ratings = pickle.load(open("data/ratings_pooled.pkl", "rb"))
        tiers = json.load(open("data/tiers.json"))
        assert set(ratings) <= set(keepers)
        assert set(tiers) <= set(keepers)

        subprocess.run([
            "jupyter", "nbconvert", "--to", "html", "--execute", str(nb_dir / "farkle_report.ipynb")
        ], check=True)
    finally:
        os.chdir(cwd)