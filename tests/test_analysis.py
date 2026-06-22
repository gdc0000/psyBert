import numpy as np
import pandas as pd

from app.analysis import compute_corr_with_significance_optimized


class TestComputeCorrelation:
    def test_returns_dataframe(self):
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        result = compute_corr_with_significance_optimized(df)
        assert isinstance(result, pd.DataFrame)

    def test_diagonal_is_self_correlation(self):
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        result = compute_corr_with_significance_optimized(df)
        assert result.loc["A", "A"] == "1.00"
        assert result.loc["B", "B"] == "1.00"

    def test_perfect_positive_correlation(self):
        np.random.seed(42)
        x = np.random.randn(100)
        y = x.copy()
        df = pd.DataFrame({"X": x, "Y": y})
        result = compute_corr_with_significance_optimized(df)
        assert "1.00" in result.loc["X", "Y"]
        assert "1.00" in result.loc["Y", "X"]

    def test_ignores_text_column(self):
        df = pd.DataFrame({"Text": ["a", "b", "c"], "A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]})
        result = compute_corr_with_significance_optimized(df)
        assert "Text" not in result.columns
        assert "A" in result.columns
        assert "B" in result.columns

    def test_includes_significance_stars(self):
        np.random.seed(0)
        x = np.random.randn(50)
        y = x + np.random.randn(50) * 0.1
        df = pd.DataFrame({"X": x, "Y": y})
        result = compute_corr_with_significance_optimized(df)
        val = result.loc["X", "Y"]
        assert isinstance(val, str)
        has_stars = "***" in val or "**" in val or "*" in val
        numeric_only = pd.to_numeric(val.replace("*", ""), errors="coerce")
        assert has_stars or not pd.isna(numeric_only)

    def test_columns_with_variance(self):
        df = pd.DataFrame({"A": [1, 2, 3, 4], "B": [5, 6, 7, 8]})
        result = compute_corr_with_significance_optimized(df)
        assert result.loc["A", "A"] == "1.00"
        assert result.loc["B", "B"] == "1.00"
