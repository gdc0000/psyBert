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
from sklearn.decomposition import PCA  # For PCA analysis

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
    else:
        return pd.read_excel(file)

@st.cache_data(show_spinner=False)
def load_scales(file) -> (dict, dict):
    """
    Load validated scales from an Excel file.
    Each sheet should contain columns "Item" and "Rev".
    Returns:
        scales_data: dict mapping construct names to a list of items.
        reverse_items_dict: dict mapping construct names to a list of computed reverse indices.
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
            sims = util.cos_sim(text_embeddings, item_embed.unsqueeze(0))  # shape: (n_texts, 1)
            sims_np = sims.cpu().numpy().flatten()
            if i in reverse_items.get(scale, []):
                sims_np = 1 - sims_np
            item_scores.append(sims_np)
        aggregated = np.mean(np.array(item_scores), axis=0)
        results[scale] = aggregated
    lengths = {key: len(val) for key, val in results.items()}
    if len(set(lengths.values())) > 1:
        st.error("Mismatch in data lengths. Please check your text data for missing values.")
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
            sims_np = sims.cpu().numpy().flatten()
            if i in reverse_items.get(scale, []):
                sims_np = 1 - sims_np
            col_name = f"{scale}_{i+1}"
            results[col_name] = sims_np
    lengths = {key: len(val) for key, val in results.items()}
    if len(set(lengths.values())) > 1:
        st.error("Mismatch in data lengths. Please check your text data for missing values.")
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
        results[name] = sims.cpu().numpy().flatten()
    lengths = {key: len(val) for key, val in results.items()}
    if len(set(lengths.values())) > 1:
        st.error("Mismatch in data lengths. Please check your text data for missing values.")
        return None
    return pd.DataFrame(results)

def exclude_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Exclude rows with z-scores greater than 3 in any similarity score column."""
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

def perform_factor_analysis(data: pd.DataFrame, analysis_type: str, n_factors: int, rotation: str, show_scree: bool):
    """
    Perform factor analysis based on the selected method.
    analysis_type: 'EFA', 'PCA', or 'CFA'
    n_factors: number of factors to extract
    rotation: for EFA (e.g., 'varimax', 'oblimin', or 'none')
    show_scree: if True, display a scree plot of eigenvalues.
    Returns a dict with results.
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
                scree_fig = px.line(x=list(range(1, len(eigenvalues)+1)), y=eigenvalues,
                                    markers=True, labels={'x': 'Factor', 'y': 'Eigenvalue'},
                                    title="Scree Plot (EFA)")
                results["scree_plot"] = scree_fig
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
                scree_fig = px.line(x=list(range(1, len(eigenvalues)+1)), y=eigenvalues,
                                    markers=True, labels={'x': 'Component', 'y': 'Eigenvalue'},
                                    title="Scree Plot (PCA)")
                results["scree_plot"] = scree_fig
        except Exception as e:
            st.error(f"PCA error: {e}")
    elif analysis_type == "CFA":
        st.error("Confirmatory Factor Analysis (CFA) is not implemented in this version.")
    else:
        st.error("Invalid factor analysis type selected.")
    return results

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
# Session State Initialization
# =============================================================================
if "constructs" not in st.session_state:
    st.session_state.constructs = []  # For method 3: interactive constructs
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

# Depending on the method, show scales interface or interactive construct input.
if scoring_method in ["Aggregated Items (Excel Upload)", "Item-by-item (Excel Upload)"]:
    st.sidebar.subheader("Upload Validated Scales (Excel)")
    scales_file = st.sidebar.file_uploader("Upload Excel file", type=["xlsx"], key="scales_file")
    if scales_file:
        try:
            all_scales_data, all_reverse_items_dict = load_scales(scales_file)
            # Provide a multiselect menu to choose which scales to include
            available_scales = list(all_scales_data.keys())
            selected_scales = st.sidebar.multiselect("Select scales to include", options=available_scales, default=available_scales, key="selected_scales")
            selected_scales_data = {}
            selected_reverse_items = {}
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
st.write("Configure file uploads and scoring method in the sidebar. Then, proceed with analysis below.")

# Step 1: Generate Text Embeddings
st.markdown("---")
st.header("Step 1: Generate Text Embeddings")
if st.button("Generate Text Embeddings", key="btn_gen_text_embed"):
    if st.session_state.text_data is None or st.session_state.text_column is None:
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
    if st.session_state.text_embeddings is None:
        st.error("Please generate text embeddings first.")
    else:
        if st.session_state.method in ["Aggregated Items (Excel Upload)", "Item-by-item (Excel Upload)"]:
            if not st.session_state.scales_data:
                st.error("Please upload validated scales (Excel file) in the sidebar.")
            else:
                with st.spinner("Computing similarity scores..."):
                    if st.session_state.method == "Aggregated Items (Excel Upload)":
                        sim_df = compute_similarity_scores_aggregated(
                            st.session_state.model_instance,
                            st.session_state.text_embeddings,
                            st.session_state.scales_data,
                            st.session_state.reverse_items
                        )
                    else:
                        sim_df = compute_similarity_scores_item_by_item(
                            st.session_state.model_instance,
                            st.session_state.text_embeddings,
                            st.session_state.scales_data,
                            st.session_state.reverse_items
                        )
        else:  # Single Construct (Interactive Input)
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

# Factor Analysis Customization Section
st.markdown("---")
st.header("Step 5: Factor Analysis Customization")
fa_type = st.selectbox("Choose factor analysis type:", options=["EFA", "PCA", "CFA"])
n_factors = st.number_input("Number of factors/components:", min_value=1, max_value=20, value=2, step=1)
rotation_method = st.selectbox("Rotation method (for EFA):", options=["varimax", "oblimin", "none"])
show_scree = st.checkbox("Show scree plot", value=True)

if st.button("Run Factor Analysis", key="btn_fa"):
    if st.session_state.similarity_results is None:
        st.error("Please compute similarity scores first.")
    else:
        fa_results = perform_factor_analysis(st.session_state.similarity_results,
                                             analysis_type=fa_type,
                                             n_factors=n_factors,
                                             rotation=rotation_method,
                                             show_scree=show_scree)
        if fa_results:
            st.subheader(f"Eigenvalues ({fa_type})")
            st.write(fa_results.get("eigenvalues", "No eigenvalues returned."))
            st.subheader(f"Loadings ({fa_type})")
            st.write(fa_results.get("loadings", "No loadings returned."))
            if show_scree and "scree_plot" in fa_results:
                st.plotly_chart(fa_results["scree_plot"], use_container_width=True)

# Step 6: Descriptive Statistics & Visualizations (Using Plotly)
st.markdown("---")
st.header("Step 6: Descriptive Statistics & Visualizations")
if st.button("Show Descriptive Statistics & Visualizations", key="btn_desc_stats"):
    if st.session_state.similarity_results is None:
        st.error("Please compute similarity scores first.")
    else:
        df = st.session_state.similarity_results.copy()
        st.subheader("Descriptive Statistics")
        st.write(df.describe())
        
        score_cols = [col for col in df.columns if col != "Text"]
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

# Step 7: Correlation Analysis
st.markdown("---")
st.header("Step 7: Correlation Analysis")
if st.button("Run Correlation Analysis", key="btn_corr_analysis"):
    if st.session_state.similarity_results is None:
        st.error("Please compute similarity scores first.")
    else:
        df = st.session_state.similarity_results.copy()
        score_cols = [col for col in df.columns if col != "Text"]
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

# Add persistent footer
add_footer()
