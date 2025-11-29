import logging

import pandas as pd

from farkle.analysis import agreement


def test_assert_no_ties_warns_on_duplicates(caplog):
    series = pd.Series([1.0, 1.0], index=["a", "b"])

    with caplog.at_level(logging.WARNING):
        agreement._assert_no_ties(series, "test scores")

    assert "Ties detected in test scores" in caplog.text
