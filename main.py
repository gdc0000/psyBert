import streamlit as st
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
import time
import logging
from scipy.stats import zscore
from factor_analyzer import FactorAnalyzer
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure logging for debugging purposes
logging.basicConfig(level=logging.INFO)

# Set page configuration
st.set_page_config(page_title="BERT-based Text Analysis Application", layout="wide")

# =============================================================================
# Helper Functions (with caching for expensive operations)
# =============================================================================

@st.cache_data(show_spinner=False)
def load_text_data(file, file_type: str) -> pd.DataFrame:
    """Load text data from CSV or Excel."""
    if file_type == "csv":
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

@st.cache_data(show_spinner=False)
def load_scales(file) -> (dict, dict):
    """
    Load validated scales from an Excel file.
    Returns:
        scales_data: Dict of scale_name -> list of items.
        reverse_items_dict: Dict of scale_name -> list of computed reverse indices.
    """
    xls = pd.ExcelFile(file)
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
    """Generate embeddings for the provided texts using the given model."""
    embeddings = []
    progress_bar = st.progress(0)
    for i, text in enumerate(texts):
        embeddings.append(model.encode(text, convert_to_tensor=True))
        progress_bar.progress((i + 1) / len(texts))
        time.sleep(0.01)  # Simulate delay for UX
    return torch.stack(embeddings)

def compute_similarity_scores(model: SentenceTransformer, text_embeddings: torch.Tensor,
                              scales_data: dict, reverse_items: dict) -> pd.DataFrame:
    """
    Compute similarity scores between text embeddings and each scale item individually.
    Returns a DataFrame with a column per scale item (e.g., "CN_1", "CN_2", ...).
    """
    results = {"Text": st.session_state.text_data[st.session_state.text_column].dropna().tolist()}
    for scale, items in scales_data.items():
        st.write(f"Processing scale: {scale}")
        item_embeds = model.encode(items, convert_to_tensor=True)
        sims = util.cos_sim(text_embeddings, item_embeds)
        if scale in reverse_items:
            rev_idx = reverse_items[scale]
            sims[:, rev_idx] = 1 - sims[:, rev_idx]
        sims_np = sims.cpu().numpy()
        for j in range(sims_np.shape[1]):
            col_name = f"{scale}_{j+1}"
            results[col_name] = sims_np[:, j]
    lengths = {key: len(val) for key, val in results.items()}
    if len(set(lengths.values())) > 1:
        st.error("Mismatch in data lengths. Please ensure your text column has no missing values.")
        return None
    return pd.DataFrame(results)

def exclude_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Exclude rows with z-scores greater than 3 in any similarity score column."""
    score_cols = [col for col in df.columns if "_" in col and col != "Text"]
    z_scores = df[score_cols].apply(zscore)
    mask = (np.abs(z_scores) <= 3).all(axis=1)
    return df[mask]

def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply Min-Max normalization to similarity score columns."""
    score_cols = [col for col in df.columns if "_" in col and col != "Text"]
    df_norm = df.copy()
    for col in score_cols:
        df_norm[col] = (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min())
    return df_norm

def add_footer() -> None:
    """Add a persistent footer to all pages."""
    st.markdown("---")
    st.markdown("### **Gabriele Di Cicco, PhD in Social Psychology**")
    st.markdown("""
    [GitHub](https://github.com/gdc0000) | 
    [ORCID](https://orcid.org/0000-0002-1439-5790) | 
    [LinkedIn](https://www.linkedin.com/in/gabriele-di-cicco-124067b0/)
    """)

# =============================================================================
# Sidebar: File Uploads and Configurations
# =============================================================================
st.sidebar.header("File Uploads and Configuration")

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

# Upload Validated Scales
scales_file = st.sidebar.file_uploader("Upload Validated Scales (Excel with sheets per construct)", type=["xlsx"], key="scales_file")
if scales_file:
    try:
        scales_data, reverse_items_dict = load_scales(scales_file)
        # For each scale, allow user to review and adjust reverse items.
        for scale, items in scales_data.items():
            st.sidebar.write(f"Review reverse items for scale: {scale}")
            df_preview = pd.DataFrame({"Index": list(range(len(items))), "Item": items})
            st.sidebar.dataframe(df_preview)
            user_rev = st.sidebar.multiselect(
                f"Select reverse item indices for {scale}",
                options=list(range(len(items))),
                default=reverse_items_dict[scale],
                key=f"rev_{scale}"
            )
            reverse_items_dict[scale] = user_rev
        st.session_state.scales_data = scales_data
        st.session_state.reverse_items = reverse_items_dict
    except Exception as e:
        st.sidebar.error(f"Error loading scales file: {e}")

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
st.write("Configure file uploads and options in the sidebar. Then, proceed with the analysis below.")

# Step 1: Generate Text Embeddings
st.markdown("---")
st.header("Step 1: Generate Text Embeddings")
if st.button("Generate Text Embeddings", key="btn_gen_text_embed"):
    if st.session_state.text_data is None or st.session_state.text_column is None:
        st.error("Please upload your text data and select the textual column in the sidebar.")
    else:
        texts = st.session_state.text_data[st.session_state.text_column].dropna().tolist()
        with st.spinner("Generating text embeddings..."):
            embeddings = generate_text_embeddings(st.session_state.model_instance, texts)
        st.session_state.text_embeddings = embeddings
        st.success("Text embeddings generated successfully.")
        st.write("Embeddings shape:", embeddings.shape)

# Step 2: Compute Similarity Scores for Scales (Item-by-Item)
st.markdown("---")
st.header("Step 2: Compute Similarity Scores for Scales (Item-by-Item)")
if st.button("Compute Similarity Scores", key="btn_compute_sim"):
    if not st.session_state.scales_data:
        st.error("Please upload the validated scales file in the sidebar.")
    elif st.session_state.text_embeddings is None:
        st.error("Please generate text embeddings first.")
    else:
        with st.spinner("Computing similarity scores..."):
            sim_df = compute_similarity_scores(
                st.session_state.model_instance,
                st.session_state.text_embeddings,
                st.session_state.scales_data,
                st.session_state.reverse_items
            )
        if sim_df is not None:
            st.session_state.similarity_results = sim_df
            st.success("Similarity scores computed and compiled.")
            st.write(sim_df.head())

# Step 3: Exclude Outliers (Z-score > 3)
st.markdown("---")
st.header("Step 3: Exclude Outliers (Z-score > 3)")
if st.button("Exclude Outliers", key="btn_exclude_outliers"):
    if st.session_state.similarity_results is None:
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
    if st.session_state.similarity_results is None:
        st.error("Please compute similarity scores first.")
    else:
        df_norm = normalize_data(st.session_state.similarity_results)
        st.session_state.normalized_df = df_norm
        st.success("Data normalized successfully.")
        st.write(df_norm.head())

# Step 5: Descriptive Statistics & Visualizations (Using Plotly)
st.markdown("---")
st.header("Step 5: Descriptive Statistics & Visualizations")
if st.button("Show Descriptive Statistics & Visualizations", key="btn_desc_stats"):
    if st.session_state.similarity_results is None:
        st.error("Please compute similarity scores first.")
    else:
        df = st.session_state.similarity_results.copy()
        st.subheader("Descriptive Statistics")
        st.write(df.describe())
        
        # Determine the score columns (excluding the "Text" column)
        score_cols = [col for col in df.columns if "_" in col and col != "Text"]
        
        # Create interactive histograms with Plotly: arrange five per row
        n_cols = 5
        n_plots = len(score_cols)
        n_rows = (n_plots + n_cols - 1) // n_cols
        fig_hist = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=score_cols)
        for idx, col in enumerate(score_cols):
            row = (idx // n_cols) + 1
            col_idx = (idx % n_cols) + 1
            hist_fig = px.histogram(df, x=col, nbins=30)
            for trace in hist_fig.data:
                fig_hist.add_trace(trace, row=row, col=col_idx)
        fig_hist.update_layout(height=300 * n_rows, width=2000, title_text="Histograms", showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Create interactive correlation heatmap with Plotly (without annotations)
        corr = df[score_cols].corr()
        heatmap_fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale='Viridis',
            showscale=True,
            hoverinfo='x+y+z'
        ))
        heatmap_fig.update_layout(title="Correlation Heatmap", xaxis_nticks=36)
        st.plotly_chart(heatmap_fig, use_container_width=True)

# Step 6: Correlation Analysis
st.markdown("---")
st.header("Step 6: Correlation Analysis")
if st.button("Run Correlation Analysis", key="btn_corr_analysis"):
    if st.session_state.similarity_results is None:
        st.error("Please compute similarity scores first.")
    else:
        df = st.session_state.similarity_results.copy()
        score_cols = [col for col in df.columns if "_" in col and col != "Text"]
        corr_matrix = df[score_cols].corr()
        st.subheader("Correlation Matrix")
        st.dataframe(corr_matrix)
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='Viridis',
            showscale=True
        ))
        fig_corr.update_layout(title="Correlation Matrix Heatmap", xaxis_nticks=36)
        st.plotly_chart(fig_corr, use_container_width=True)

# Step 7: Exploratory Factor Analysis (EFA)
st.markdown("---")
st.header("Step 7: Exploratory Factor Analysis (EFA)")
if st.button("Run Exploratory Factor Analysis (EFA)", key="btn_efa"):
    if st.session_state.similarity_results is None:
        st.error("Please compute similarity scores first.")
    else:
        import scipy
        scipy.sum = np.sum  # Monkey patch for compatibility with Python 3.12+
        df = st.session_state.similarity_results.copy()
        score_cols = [col for col in df.columns if "_" in col and col != "Text"]
        data_for_efa = df[score_cols]
        try:
            fa = FactorAnalyzer(n_factors=2, rotation="varimax")
            fa.fit(data_for_efa)
            eigenvalues, _ = fa.get_eigenvalues()
            loadings = pd.DataFrame(fa.loadings_, index=score_cols)
            st.subheader("Eigenvalues")
            st.write(eigenvalues)
            st.subheader("Factor Loadings")
            st.write(loadings)
        except Exception as e:
            st.error(f"Error during factor analysis: {e}")

st.markdown("---")
st.write("Session State Keys:", list(st.session_state.keys()))

# Add persistent footer
add_footer()
