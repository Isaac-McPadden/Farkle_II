import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from farkle.run_rf import plot_partial_dependence


def test_plot_partial_dependence(tmp_path):
    X = pd.DataFrame({"a": range(5), "b": range(5, 10)})
    y = pd.Series(range(5))
    model = HistGradientBoostingRegressor(random_state=0)
    model.fit(X, y)

    out_file = plot_partial_dependence(model, X, "a", tmp_path)
    assert out_file.exists()
