import streamlit as st
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
import time
import logging
from scipy.stats import zscore, pearsonr
from factor_analyzer import FactorAnalyzer
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
import os

# Disable CodeCarbon tracking in transformers (if needed)
os.environ["TRANSFORMERS_NO_CODECARBON"] = "1"

# Configure logging
logging.basicConfig(level=logging.INFO)

# Set page configuration
st.set_page_config(page_title="BERT-based Text Analysis Application", layout="wide")

# -----------------------------------------------------------------------------
# Session State Initialization for uploaded data and parameters
# -----------------------------------------------------------------------------
if "scales_data" not in st.session_state:
    st.session_state.scales_data = {}
if "reverse_items" not in st.session_state:
    st.session_state.reverse_items = {}
if "constructs" not in st.session_state:
    st.session_state.constructs = []  # For method 3: interactive constructs
if "similarity_results" not in st.session_state:
    st.session_state.similarity_results = None

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_text_data(file_bytes, file_type: str) -> pd.DataFrame:
    """Load text data from CSV or Excel using file bytes."""
    if file_type == "csv":
        from io import StringIO
        return pd.read_csv(StringIO(file_bytes.decode('utf-8')))
    else:
        from io import BytesIO
        return pd.read_excel(BytesIO(file_bytes))

@st.cache_data(show_spinner=False)
def load_scales(file_bytes) -> (dict, dict):
    """
    Load validated scales from an Excel file.
    Each sheet should contain columns "Item" and "Rev".
    Returns:
        scales_data: dict mapping construct names to list of items.
        reverse_items_dict: dict mapping construct names to list of computed reverse indices.
    """
    from io import BytesIO
    xls = pd.ExcelFile(BytesIO(file_bytes))
    scales_data = {}
    reverse_items_dict = {}
    for sheet_name in xls.sheet_names:
        df_sheet = pd.read_excel(xls, sheet_name=sheet_name)
        if "Item" in df_sheet.columns and "Rev" in df_sheet.columns:
            df_sheet = df_sheet.dropna(subset=["Item"])
            items = df_sheet["Item"].tolist()
            try:
                computed_rev = [i for i, val in enumerate(df_sheet["Rev"].tolist()) if float(val) == 1.0]
            except Exception as e:
                st.sidebar.error(f"Error processing reverse items in sheet '{sheet_name}': {e}")
                computed_rev = []
            scales_data[sheet_name] = items
            reverse_items_dict[sheet_name] = computed_rev
        else:
            st.sidebar.error(f"Sheet '{sheet_name}' must have both 'Item' and 'Rev' columns.")
    return scales_data, reverse_items_dict

@st.cache_resource(show_spinner=False)
def get_model(model_name: str) -> SentenceTransformer:
    """Load and return the SentenceTransformer model."""
    return SentenceTransformer(model_name)

def generate_text_embeddings(model: SentenceTransformer, texts: list) -> torch.Tensor:
    """Generate embeddings for provided texts."""
    embeddings = []
    progress_bar = st.progress(0)
    for i, text in enumerate(texts):
        embeddings.append(model.encode(text, convert_to_tensor=True))
        progress_bar.progress((i + 1) / len(texts))
        time.sleep(0.01)
    return torch.stack(embeddings)

def compute_similarity_scores_aggregated(model: SentenceTransformer, text_embeddings: torch.Tensor,
                                           scales_data: dict, reverse_items: dict) -> pd.DataFrame:
    """
    Method 1: For each construct (Excel sheet), embed each item,
    apply reverse scoring if needed, and average the similarity scores.
    Returns one similarity score per construct.
    """
    results = {"Text": st.session_state.text_data[st.session_state.text_column].dropna().tolist()}
    for scale, items in scales_data.items():
        item_scores = []
        for i, item in enumerate(items):
            item_embed = model.encode(item, convert_to_tensor=True)
            sims = util.cos_sim(text_embeddings, item_embed.unsqueeze(0))
            # Instead of sims.cpu().numpy(), use conversion via list
            sims_np = np.array(sims.cpu().tolist()).flatten()
            if i in reverse_items.get(scale, []):
                sims_np = 1 - sims_np
            item_scores.append(sims_np)
        aggregated = np.mean(np.array(item_scores), axis=0)
        results[scale] = aggregated
    lengths = {key: len(val) for key, val in results.items()}
    if len(set(lengths.values())) > 1:
        st.error("Mismatch in data lengths. Check your text data for missing values.")
        return None
    return pd.DataFrame(results)

def compute_similarity_scores_item_by_item(model: SentenceTransformer, text_embeddings: torch.Tensor,
                                             scales_data: dict, reverse_items: dict) -> pd.DataFrame:
    """
    Method 2: For each construct (Excel sheet), embed each item separately,
    apply reverse scoring if needed, and output a separate similarity score for each item.
    """
    results = {"Text": st.session_state.text_data[st.session_state.text_column].dropna().tolist()}
    for scale, items in scales_data.items():
        for i, item in enumerate(items):
            item_embed = model.encode(item, convert_to_tensor=True)
            sims = util.cos_sim(text_embeddings, item_embed.unsqueeze(0))
            sims_np = np.array(sims.cpu().tolist()).flatten()
            if i in reverse_items.get(scale, []):
                sims_np = 1 - sims_np
            col_name = f"{scale}_{i+1}"
            results[col_name] = sims_np
    lengths = {key: len(val) for key, val in results.items()}
    if len(set(lengths.values())) > 1:
        st.error("Mismatch in data lengths. Check your text data for missing values.")
        return None
    return pd.DataFrame(results)

def compute_similarity_scores_single(model: SentenceTransformer, text_embeddings: torch.Tensor,
                                       constructs: list) -> pd.DataFrame:
    """
    Method 3: For each construct added interactively, embed the entire construct text as one block.
    Returns one similarity score per construct.
    """
    results = {"Text": st.session_state.text_data[st.session_state.text_column].dropna().tolist()}
    for construct in constructs:
        name = construct["name"]
        text = construct["text"]
        construct_embed = model.encode(text, convert_to_tensor=True).unsqueeze(0)
        sims = util.cos_sim(text_embeddings, construct_embed)
        results[name] = np.array(sims.cpu().tolist()).flatten()
    lengths = {key: len(val) for key, val in results.items()}
    if len(set(lengths.values())) > 1:
        st.error("Mismatch in data lengths. Check your text data for missing values.")
        return None
    return pd.DataFrame(results)

def exclude_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Exclude rows with z-scores greater than 3."""
    score_cols = [col for col in df.columns if col != "Text"]
    z_scores = df[score_cols].apply(zscore)
    mask = (np.abs(z_scores) <= 3).all(axis=1)
    return df[mask]

def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply Min-Max normalization to similarity score columns."""
    score_cols = [col for col in df.columns if col != "Text"]
    df_norm = df.copy()
    for col in score_cols:
        df_norm[col] = (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min())
    return df_norm

def compute_corr_with_significance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the correlation matrix with significance levels.
    Significance is marked as:
      * p < 0.05: *
      * p < 0.01: **
      * p < 0.001: ***
    Returns a DataFrame with strings (e.g., "0.75***").
    """
    score_cols = [col for col in df.columns if col != "Text"]
    corr_mat = df[score_cols].corr()
    n = len(score_cols)
    annotated = pd.DataFrame(index=score_cols, columns=score_cols)
    for i in range(n):
        for j in range(n):
            r = corr_mat.iloc[i, j]
            if i == j:
                annotated.iloc[i, j] = f"{r:.2f}"
            else:
                _, p = pearsonr(df[score_cols[i]], df[score_cols[j]])
                if p < 0.001:
                    stars = "***"
                elif p < 0.01:
                    stars = "**"
                elif p < 0.05:
                    stars = "*"
                else:
                    stars = ""
                annotated.iloc[i, j] = f"{r:.2f}{stars}"
    return annotated

def perform_factor_analysis(data: pd.DataFrame, analysis_type: str, n_factors: int, rotation: str, show_scree: bool):
    """
    Perform factor analysis based on the selected method.
    analysis_type: 'EFA', 'PCA', or 'CFA'
    n_factors: number of factors/components to extract
    rotation: for EFA ('varimax', 'oblimin', or 'none')
    show_scree: if True, include a scree plot.
    Returns a dict with eigenvalues, loadings, and optionally a scree plot.
    """
    results = {}
    score_cols = [col for col in data.columns if col != "Text"]
    X = data[score_cols].values

    if analysis_type == "EFA":
        try:
            fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation if rotation != "none" else None)
            fa.fit(X)
            eigenvalues, _ = fa.get_eigenvalues()
            loadings = p
