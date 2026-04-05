import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import t


@st.cache_data(show_spinner=False)
def compute_corr_with_significance_optimized(df: pd.DataFrame) -> pd.DataFrame:
    score_cols = [col for col in df.columns if col != "Text"]
    x_vals = df[score_cols].values
    n_rows = x_vals.shape[0]
    corr = np.corrcoef(x_vals, rowvar=False)
    t_stats = corr * np.sqrt((n_rows - 2) / (1 - corr**2 + np.eye(len(score_cols))))
    p_values = 2 * (1 - t.cdf(np.abs(t_stats), df=n_rows - 2))
    np.fill_diagonal(p_values, 0)

    annotated = pd.DataFrame(index=score_cols, columns=score_cols)
    for i, col_i in enumerate(score_cols):
        for j, col_j in enumerate(score_cols):
            r_val = corr[i, j]
            p_val = p_values[i, j]
            if i == j:
                annotated.loc[col_i, col_j] = f"{r_val:.2f}"
            else:
                if p_val < 0.001:
                    stars = "***"
                elif p_val < 0.01:
                    stars = "**"
                elif p_val < 0.05:
                    stars = "*"
                else:
                    stars = ""
                annotated.loc[col_i, col_j] = f"{r_val:.2f}{stars}"
    return annotated

