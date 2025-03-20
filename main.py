import sys
# Patch for Python 3.12: if 'distutils.dir_util' is missing, inject it from setuptools.
try:
    import distutils.dir_util
except ModuleNotFoundError:
    from setuptools._distutils import dir_util as distutils_dir_util
    sys.modules["distutils.dir_util"] = distutils_dir_util

import time
import logging

import streamlit as st
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from scipy.stats import zscore, pearsonr
from factor_analyzer import FactorAnalyzer
from sklearn.decomposition import PCA

# Configure logging
logging.basicConfig(level=logging.INFO)

# Set page configuration
st.set_page_config(page_title="BERT-based Text Analysis Application", layout="wide")

# =============================================================================
# Helper Functions
# =============================================================================

@st.cache_data(show_spinner=False)
def load_text_data(file, file_type: str) -> pd.DataFrame:
    """Load text data from CSV or Excel."""
    if file_type == "csv":
        return pd.read_csv(file)
    return pd.read_excel(file)

@st.cache_data(show_spinner=False)
def load_scales(file) -> (dict, dict):
    """
    Load validated scales from an Excel file.
    Each sheet should contain columns "Item" and "Rev".
    Returns:
        scales_data: dict mapping construct names to list of items.
        reverse_items_dict: dict mapping construct names to list of reverse item indices.
    """
    xls = pd.ExcelFile(file)
    scales_data, reverse_items_dict = {}, {}
    for sheet_name in xls.sheet_names:
        df_sheet = pd.read_excel(xls, sheet_name=sheet_name)
        if {"Item", "Rev"}.issubset(df_sheet.columns):
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
    """Generate embeddings for provided texts with progress feedback."""
    embeddings = []
    progress_bar = st.progress(0)
    total = len(texts)
    for i, text in enumerate(texts):
        embeddings.append(model.encode(text, convert_to_tensor=True))
        progress_bar.progress((i + 1) / total)
        time.sleep(0.01)
    return torch.stack(embeddings)

def compute_similarity_scores_aggregated(model: SentenceTransformer,
                                         text_embeddings: torch.Tensor,
                                         scales_data: dict,
                                         reverse_items: dict) -> pd.DataFrame:
    """
    For each construct (Excel sheet), embed each item, apply reverse scoring if needed,
    and average the similarity scores. Returns one similarity score per construct.
    """
    texts = st.session_state.text_data[st.session_state.text_column].dropna().tolist()
    results = {"Text": texts}
    for scale, items in scales_data.items():
        item_scores = []
        for i, item in enumerate(items):
            item_embed = model.encode(item, convert_to_tensor=True)
            sims = util.cos_sim(text_embeddings, item_embed.unsqueeze(0))
            sims_np = sims.cpu().numpy().flatten()
            if i in reverse_items.get(scale, []):
                sims_np = 1 - sims_np
            item_scores.append(sims_np)
        results[scale] = np.mean(np.array(item_scores), axis=0)
    if len({len(v) for v in results.values()}) != 1:
        st.error("Mismatch in data lengths. Check your text data for missing values.")
        return None
    return pd.DataFrame(results)

def compute_similarity_scores_item_by_item(model: SentenceTransformer,
                                           text_embeddings: torch.Tensor,
                                           scales_data: dict,
                                           reverse_items: dict) -> pd.DataFrame:
    """
    For each construct (Excel sheet), embed each item separately, apply reverse scoring if needed,
    and output a separate similarity score for each item.
    """
    texts = st.session_state.text_data[st.session_state.text_column].dropna().tolist()
    results = {"Text": texts}
    for scale, items in scales_data.items():
        for i, item in enumerate(items):
            item_embed = model.encode(item, convert_to_tensor=True)
            sims = util.cos_sim(text_embeddings, item_embed.unsqueeze(0))
            sims_np = sims.cpu().numpy().flatten()
            if i in reverse_items.get(scale, []):
                sims_np = 1 - sims_np
            results[f"{scale}_{i+1}"] = sims_np
    if len({len(v) for v in results.values()}) != 1:
        st.error("Mismatch in data lengths. Check your text data for missing values.")
        return None
    return pd.DataFrame(results)

def compute_similarity_scores_single(model: SentenceTransformer,
                                     text_embeddings: torch.Tensor,
                                     constructs: list) -> pd.DataFrame:
    """
    For each interactively added construct, embed the entire construct text as one block.
    Returns one similarity score per construct.
    """
    texts = st.session_state.text_data[st.session_state.text_column].dropna().tolist()
    results = {"Text": texts}
    for construct in constructs:
        name = construct["name"]
        construct_embed = model.encode(construct["text"], convert_to_tensor=True).unsqueeze(0)
        sims = util.cos_sim(text_embeddings, construct_embed)
        results[name] = sims.cpu().numpy().flatten()
    if len({len(v) for v in results.values()}) != 1:
        st.error("Mismatch in data lengths. Check your text data for missing values.")
        return None
    return pd.DataFrame(results)

def exclude_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Exclude rows with any z-score greater than 3 for similarity scores."""
    score_cols = [col for col in df.columns if col != "Text"]
    z_scores = df[score_cols].apply(zscore)
    return df[(np.abs(z_scores) <= 3).all(axis=1)]

def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply Min-Max normalization to similarity score columns."""
    df_norm = df.copy()
    for col in [c for c in df.columns if c != "Text"]:
        df_norm[col] = (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min())
    return df_norm

def compute_corr_with_significance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the correlation matrix with significance levels.
    Significance is marked as:
      * p < 0.05: *
      * p < 0.01: **
      * p < 0.001: ***
    Returns a DataFrame with annotations (e.g., "0.75***").
    """
    score_cols = [col for col in df.columns if col != "Text"]
    corr_mat = df[score_cols].corr()
    annotated = pd.DataFrame(index=score_cols, columns=score_cols)
    for i, col_i in enumerate(score_cols):
        for j, col_j in enumerate(score_cols):
            r = corr_mat.loc[col_i, col_j]
            if i == j:
                annotated.loc[col_i, col_j] = f"{r:.2f}"
            else:
                _, p = pearsonr(df[col_i], df[col_j])
                stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                annotated.loc[col_i, col_j] = f"{r:.2f}{stars}"
    return annotated

def perform_factor_analysis(data: pd.DataFrame, analysis_type: str, n_factors: int,
                            rotation: str, show_scree: bool) -> dict:
    """
    Perform factor analysis based on the selected method.
    analysis_type: 'EFA', 'PCA', or 'CFA'
    n_factors: number of factors/components to extract
    rotation: for EFA ('varimax', 'oblimin', or 'none')
    show_scree: if True, include a scree plot placeholder.
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
            loadings = pd.DataFrame(fa.loadings_, index=score_cols)
            results["eigenvalues"] = eigenvalues
            results["loadings"] = loadings
            if show_scree:
                results["scree_plot"] = None  # Placeholder for scree plot
        except Exception as e:
            st.error(f"EFA error: {e}")
    elif analysis_type == "PCA":
        try:
            pca = PCA(n_components=n_factors)
            pca.fit(X)
            eigenvalues = pca.explained_variance_
            loadings = pd.DataFrame(pca.components_.T, index=score_cols,
                                    columns=[f"PC{i+1}" for i in range(n_factors)])
            results["eigenvalues"] = eigenvalues
            results["loadings"] = loadings
            if show_scree:
                results["scree_plot"] = None  # Placeholder for scree plot
        except Exception as e:
            st.error(f"PCA error: {e}")
    elif analysis_type == "CFA":
        st.error("Confirmatory Factor Analysis (CFA) is not implemented.")
    else:
        st.error("Invalid factor analysis type selected.")
    return results

def add_footer() -> None:
    """Add a persistent footer to all pages."""
    st.markdown("---")
    st.markdown("### **Gabriele Di Cicco, PhD in Social Psychology**")
    st.markdown(
        "[GitHub](https://github.com/gdc0000) | "
        "[ORCID](https://orcid.org/0000-0002-1439-5790) | "
        "[LinkedIn](https://www.linkedin.com/in/gabriele-di-cicco-124067b0/)"
    )

# =============================================================================
# Session State Initialization
# =============================================================================
if "constructs" not in st.session_state:
    st.session_state.constructs = []  # For interactive constructs
if "similarity_results" not in st.session_state:
    st.session_state.similarity_results = None

# =============================================================================
# Sidebar: File Uploads and Configurations
# =============================================================================
st.sidebar.header("Configuration")

# Scoring Method Selection
scoring_method = st.sidebar.radio(
    "Choose scoring method:",
    options=[
        "Aggregated Items (Excel Upload)",
        "Item-by-item (Excel Upload)",
        "Single Construct (Interactive Input)"
    ],
    key="scoring_method"
)
st.session_state.method = scoring_method

# Upload Text Data
text_file = st.sidebar.file_uploader("Upload Text Data (CSV or Excel)", type=["csv", "xlsx"], key="text_file")
if text_file:
    file_type = "csv" if text_file.name.endswith("csv") else "xlsx"
    try:
        st.session_state.text_data = load_text_data(text_file, file_type)
        st.sidebar.write("Preview of Text Data:")
        st.sidebar.dataframe(st.session_state.text_data.head())
        cols = st.session_state.text_data.columns.tolist()
        st.session_state.text_column = st.sidebar.selectbox("Select the textual column", cols)
    except Exception as e:
        st.sidebar.error(f"Error loading text file: {e}")

# Scales Upload or Interactive Constructs
if scoring_method in ["Aggregated Items (Excel Upload)", "Item-by-item (Excel Upload)"]:
    st.sidebar.subheader("Upload Validated Scales (Excel)")
    scales_file = st.sidebar.file_uploader("Upload Excel file", type=["xlsx"], key="scales_file")
    if scales_file:
        try:
            all_scales_data, all_reverse_items_dict = load_scales(scales_file)
            available_scales = list(all_scales_data.keys())
            selected_scales = st.sidebar.multiselect("Select scales to include", options=available_scales,
                                                     default=available_scales, key="selected_scales")
            selected_scales_data, selected_reverse_items = {}, {}
            for scale in selected_scales:
                items = all_scales_data[scale]
                st.sidebar.write(f"Review reverse items for: {scale}")
                df_preview = pd.DataFrame({"Index": list(range(len(items))), "Item": items})
                st.sidebar.dataframe(df_preview)
                user_rev = st.sidebar.multiselect(
                    f"Select reverse item indices for {scale}",
                    options=list(range(len(items))),
                    default=all_reverse_items_dict[scale],
                    key=f"rev_{scale}"
                )
                selected_scales_data[scale] = items
                selected_reverse_items[scale] = user_rev
            st.session_state.scales_data = selected_scales_data
            st.session_state.reverse_items = selected_reverse_items
        except Exception as e:
            st.sidebar.error(f"Error loading scales file: {e}")
else:
    st.sidebar.subheader("Constructs (Interactive Input)")
    with st.sidebar.expander("Add a Construct", expanded=True):
        construct_name = st.text_input("Construct Name", key="construct_name")
        construct_text = st.text_area("Construct Text (paste entire construct text)", key="construct_text")
        if st.button("Add Construct", key="btn_add_construct"):
            if construct_name and construct_text:
                st.session_state.constructs.append({"name": construct_name, "text": construct_text})
                st.success(f"Construct '{construct_name}' added.")
            else:
                st.error("Please provide both name and text.")
    with st.sidebar.expander("Manage Constructs", expanded=True):
        if st.session_state.constructs:
            construct_names = [c["name"] for c in st.session_state.constructs]
            constructs_to_remove = st.multiselect("Select constructs to remove", construct_names, key="remove_constructs")
            if st.button("Remove Selected Constructs", key="btn_remove_constructs"):
                st.session_state.constructs = [c for c in st.session_state.constructs if c["name"] not in constructs_to_remove]
                st.success("Selected constructs removed.")
                st.experimental_rerun()
        else:
            st.info("No constructs added yet.")
    if st.session_state.constructs:
        st.sidebar.write("Current Constructs:")
        for c in st.session_state.constructs:
            st.sidebar.write(f"- **{c['name']}**")

# Model Selection
st.sidebar.subheader("Select Embedding Model")
model_options = {
    "all-MiniLM-L6-v2": "Lightweight and efficient model for sentence embeddings.",
    "paraphrase-MiniLM-L3-v2": "Faster model with lower dimensionality, ideal for paraphrase tasks.",
    "all-distilroberta-v1": "Robust model based on DistilRoBERTa for diverse tasks."
}
selected_model = st.sidebar.selectbox("Choose model", list(model_options.keys()), index=0)
st.sidebar.write(model_options[selected_model])
st.session_state.selected_model = selected_model

if st.session_state.get("model_instance") is None:
    with st.spinner("Loading embedding model..."):
        st.session_state.model_instance = get_model(st.session_state.selected_model)
    st.sidebar.success("Model loaded.")

# =============================================================================
# Main Window: Analysis Steps and Outputs
# =============================================================================
st.title("BERT-based Text Analysis Application")
st.markdown("### Analysis Steps")
st.write("Configure file uploads, scoring method, and analysis parameters in the sidebar. Then, proceed below.")

# Step 1: Generate Text Embeddings
st.markdown("---")
st.header("Step 1: Generate Text Embeddings")
if st.button("Generate Text Embeddings", key="btn_gen_text_embed"):
    # Check explicitly for None to avoid ambiguity with DataFrame truth value
    if st.session_state.get("text_data") is None or st.session_state.get("text_column") is None:
        st.error("Please upload text data and select the textual column in the sidebar.")
    else:
        texts = st.session_state.text_data[st.session_state.text_column].dropna().tolist()
        with st.spinner("Generating text embeddings..."):
            embeddings = generate_text_embeddings(st.session_state.model_instance, texts)
        st.session_state.text_embeddings = embeddings
        st.success("Text embeddings generated successfully.")
        st.write("Embeddings shape:", embeddings.shape)

# Step 2: Compute Similarity Scores
st.markdown("---")
st.header("Step 2: Compute Similarity Scores")
if st.button("Compute Similarity Scores", key="btn_compute_sim"):
    if st.session_state.get("text_embeddings") is None:
        st.error("Please generate text embeddings first.")
    else:
        if st.session_state.method in ["Aggregated Items (Excel Upload)", "Item-by-item (Excel Upload)"]:
            if not st.session_state.get("scales_data"):
                st.error("Please upload validated scales (Excel file) in the sidebar.")
            else:
                with st.spinner("Computing similarity scores..."):
                    sim_df = (
                        compute_similarity_scores_aggregated(
                            st.session_state.model_instance,
                            st.session_state.text_embeddings,
                            st.session_state.scales_data,
                            st.session_state.reverse_items
                        )
                        if st.session_state.method == "Aggregated Items (Excel Upload)"
                        else compute_similarity_scores_item_by_item(
                            st.session_state.model_instance,
                            st.session_state.text_embeddings,
                            st.session_state.scales_data,
                            st.session_state.reverse_items
                        )
                    )
        else:
            if not st.session_state.constructs:
                st.error("Please add at least one construct in the sidebar.")
            else:
                with st.spinner("Computing similarity scores..."):
                    sim_df = compute_similarity_scores_single(
                        st.session_state.model_instance,
                        st.session_state.text_embeddings,
                        st.session_state.constructs
                    )
        if sim_df is not None:
            st.session_state.similarity_results = sim_df
            st.success("Similarity scores computed and compiled.")
            st.write(sim_df.head())

# Step 3: Exclude Outliers (Z-score > 3)
st.markdown("---")
st.header("Step 3: Exclude Outliers (Z-score > 3)")
if st.button("Exclude Outliers", key="btn_exclude_outliers"):
    if st.session_state.get("similarity_results") is None:
        st.error("Please compute similarity scores first.")
    else:
        df_no_outliers = exclude_outliers(st.session_state.similarity_results)
        st.session_state.similarity_results = df_no_outliers
        st.success("Outliers excluded.")
        st.write("Data shape after outlier removal:", df_no_outliers.shape)
        st.write(df_no_outliers.head())

# Step 4: Normalize Data (Min-Max Normalization)
st.markdown("---")
st.header("Step 4: Normalize Data (Min-Max Normalization)")
if st.button("Normalize Data", key="btn_normalize_data"):
    if st.session_state.get("similarity_results") is None:
        st.error("Please compute similarity scores first.")
    else:
        df_norm = normalize_data(st.session_state.similarity_results)
        st.session_state.normalized_df = df_norm
        st.success("Data normalized successfully.")
        st.write(df_norm.head())

# Step 5: Descriptive & Correlation Analysis
st.markdown("---")
st.header("Step 5: Descriptive & Correlation Analysis")
if st.button("Show Descriptives & Correlation Table", key="btn_corr_table"):
    if st.session_state.get("similarity_results") is None:
        st.error("Please compute similarity scores first.")
    else:
        df = st.session_state.similarity_results.copy()
        st.subheader("Descriptive Statistics")
        st.write(df.describe())
        st.subheader("Correlation Matrix with Significance")
        corr_annotated = compute_corr_with_significance(df)
        st.dataframe(corr_annotated)

# Step 6: Download Enhanced Dataset
st.markdown("---")
st.header("Step 6: Download Enhanced Dataset")
if st.session_state.get("similarity_results") is not None:
    download_df = st.session_state.get("normalized_df", st.session_state.similarity_results)
    csv = download_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name='enhanced_dataset.csv',
        mime='text/csv'
    )

# Persistent Footer
add_footer()
