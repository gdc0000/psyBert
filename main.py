import streamlit as st
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from factor_analyzer import FactorAnalyzer
import time

# Set page configuration
st.set_page_config(page_title="BERT-based Text Analysis Application", layout="wide")

# Initialize session state variables if not already set
if "text_data" not in st.session_state:
    st.session_state.text_data = None
if "text_column" not in st.session_state:
    st.session_state.text_column = None
if "text_embeddings" not in st.session_state:
    st.session_state.text_embeddings = None
if "scales_data" not in st.session_state:
    st.session_state.scales_data = {}
if "reverse_items" not in st.session_state:
    st.session_state.reverse_items = {}
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "all-MiniLM-L6-v2"
if "model_instance" not in st.session_state:
    st.session_state.model_instance = None
if "similarity_results" not in st.session_state:
    st.session_state.similarity_results = None
if "normalized_df" not in st.session_state:
    st.session_state.normalized_df = None

# =============================================================================
# Sidebar: File Uploads and Configuration
# =============================================================================
st.sidebar.header("File Uploads and Configuration")

# --- Upload Text Data ---
st.sidebar.subheader("Upload Text Data")
text_file = st.sidebar.file_uploader("Upload Text Data (CSV or Excel)", type=["csv", "xlsx"], key="text_file")
if text_file:
    try:
        if text_file.name.endswith("csv"):
            df_text = pd.read_csv(text_file)
        else:
            df_text = pd.read_excel(text_file)
        st.session_state.text_data = df_text
        st.sidebar.write("Preview of Text Data:")
        st.sidebar.dataframe(df_text.head())
        # Allow user to choose the textual column
        cols = df_text.columns.tolist()
        selected_column = st.sidebar.selectbox("Select the textual column", cols)
        st.session_state.text_column = selected_column
    except Exception as e:
        st.sidebar.error(f"Error reading text file: {e}")

# --- Upload Validated Scales ---
st.sidebar.subheader("Upload Validated Scales")
scales_file = st.sidebar.file_uploader("Upload Validated Scales (Excel with sheets per construct)", type=["xlsx"], key="scales_file")
if scales_file:
    try:
        xls = pd.ExcelFile(scales_file)
        scales_data = {}
        reverse_items_dict = {}
        for sheet_name in xls.sheet_names:
            df_sheet = pd.read_excel(xls, sheet_name=sheet_name)
            # Check for required columns "Item" and "Rev"
            if "Item" in df_sheet.columns and "Rev" in df_sheet.columns:
                # Drop rows with missing items
                df_sheet = df_sheet.dropna(subset=["Item"])
                items = df_sheet["Item"].tolist()
                # Compute reverse indices: rows with Rev == 1 (as float)
                try:
                    computed_rev = [i for i, val in enumerate(df_sheet["Rev"].tolist()) if float(val) == 1.0]
                except Exception as e:
                    computed_rev = []
                    st.sidebar.error(f"Error processing reverse items in sheet '{sheet_name}': {e}")
                # Show a preview table of items with indices for user verification
                st.sidebar.write(f"Review reverse items for scale: {sheet_name}")
                df_preview = pd.DataFrame({
                    "Index": list(range(len(items))),
                    "Item": items,
                    "Rev": df_sheet["Rev"].tolist()
                })
                st.sidebar.dataframe(df_preview)
                # Let the user adjust the reverse indices using multiselect (default = computed_rev)
                user_rev = st.sidebar.multiselect(
                    f"Select reverse item indices for {sheet_name}",
                    options=list(range(len(items))),
                    default=computed_rev,
                    key=f"rev_{sheet_name}"
                )
                scales_data[sheet_name] = items
                reverse_items_dict[sheet_name] = user_rev
            else:
                st.sidebar.error(f"Sheet '{sheet_name}' must have both 'Item' and 'Rev' columns.")
        st.session_state.scales_data = scales_data
        st.session_state.reverse_items = reverse_items_dict
    except Exception as e:
        st.sidebar.error(f"Error reading scales file: {e}")

# --- Model Selection ---
st.sidebar.subheader("Select Embedding Model")
model_options = {
    "all-MiniLM-L6-v2": "Lightweight and efficient model for sentence embeddings.",
    "paraphrase-MiniLM-L3-v2": "Faster model with lower dimensionality, ideal for paraphrase tasks.",
    "all-distilroberta-v1": "Robust model based on DistilRoBERTa for diverse tasks."
}
selected_model = st.sidebar.selectbox("Choose model", list(model_options.keys()), index=0)
st.sidebar.write(model_options[selected_model])
st.session_state.selected_model = selected_model

# Load the selected model if not already loaded
if st.session_state.model_instance is None:
    with st.spinner("Loading embedding model..."):
        st.session_state.model_instance = SentenceTransformer(st.session_state.selected_model)
    st.sidebar.success("Model loaded.")

# =============================================================================
# Main Window: Analysis Steps and Outputs
# =============================================================================
st.title("BERT-based Text Analysis Application")
st.markdown("### Analysis Steps")
st.write("Configure file uploads and options in the sidebar. Then, proceed with the analysis below.")

st.markdown("---")
st.header("Step 1: Generate Text Embeddings")
if st.button("Generate Text Embeddings", key="btn_gen_text_embed"):
    if st.session_state.text_data is None or st.session_state.text_column is None:
        st.error("Please upload your text data and select the textual column in the sidebar.")
    else:
        # Use only non-null values for consistency
        texts = st.session_state.text_data[st.session_state.text_column].dropna().tolist()
        embeddings = []
        progress_bar = st.progress(0)
        st.info("Generating embeddings for textual data...")
        for i, text in enumerate(texts):
            embedding = st.session_state.model_instance.encode(text, convert_to_tensor=True)
            embeddings.append(embedding)
            progress_bar.progress((i + 1) / len(texts))
            time.sleep(0.01)  # Simulate delay
        st.session_state.text_embeddings = torch.stack(embeddings)
        st.success("Text embeddings generated successfully.")
        st.write("Embeddings shape:", st.session_state.text_embeddings.shape)

st.markdown("---")
st.header("Step 2: Compute Similarity Scores for Scales (Item-by-Item)")
if st.button("Compute Similarity Scores", key="btn_compute_sim"):
    if not st.session_state.scales_data:
        st.error("Please upload the validated scales file in the sidebar.")
    elif st.session_state.text_embeddings is None:
        st.error("Please generate text embeddings first.")
    else:
        texts = st.session_state.text_data[st.session_state.text_column].dropna().tolist()
        results = {"Text": texts}
        for scale, items in st.session_state.scales_data.items():
            st.write(f"Processing scale: {scale}")
            # Compute embeddings for each scale item
            item_embeds = st.session_state.model_instance.encode(items, convert_to_tensor=True)
            sims = util.cos_sim(st.session_state.text_embeddings, item_embeds)
            # Apply reverse scoring using the (possibly adjusted) reverse indices
            if scale in st.session_state.reverse_items:
                rev_idx = st.session_state.reverse_items[scale]
                sims[:, rev_idx] = 1 - sims[:, rev_idx]
            sims_np = sims.cpu().numpy()  # shape: (n_texts, n_items)
            # Instead of aggregating, output each item separately.
            for j in range(sims_np.shape[1]):
                col_name = f"{scale}_{j+1}"
                results[col_name] = sims_np[:, j]
        # Debug: Check lengths to ensure consistency
        lengths = {key: len(val) for key, val in results.items()}
        st.write("Array lengths in results:", lengths)
        if len(set(lengths.values())) > 1:
            st.error("Mismatch in data lengths. Please ensure your text column has no missing values.")
        else:
            st.session_state.similarity_results = pd.DataFrame(results)
            st.success("Similarity scores computed and compiled.")
            st.write(st.session_state.similarity_results.head())

st.markdown("---")
st.header("Step 3: Exclude Outliers (Z-score > 3)")
if st.button("Exclude Outliers", key="btn_exclude_outliers"):
    if st.session_state.similarity_results is None:
        st.error("Please compute similarity scores first.")
    else:
        df = st.session_state.similarity_results.copy()
        score_cols = [col for col in df.columns if "_" in col and col != "Text"]
        z_scores = df[score_cols].apply(zscore)
        mask = (np.abs(z_scores) <= 3).all(axis=1)
        df_no_outliers = df[mask]
        st.session_state.similarity_results = df_no_outliers
        st.success("Outliers excluded.")
        st.write("Data shape after outlier removal:", df_no_outliers.shape)
        st.write(st.session_state.similarity_results.head())

st.markdown("---")
st.header("Step 4: Normalize Data (Min-Max Normalization)")
if st.button("Normalize Data", key="btn_normalize_data"):
    if st.session_state.similarity_results is None:
        st.error("Please compute similarity scores first.")
    else:
        df = st.session_state.similarity_results.copy()
        score_cols = [col for col in df.columns if "_" in col and col != "Text"]
        df_norm = df.copy()
        for col in score_cols:
            df_norm[col] = (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min())
        st.session_state.normalized_df = df_norm
        st.success("Data normalized successfully.")
        st.write(st.session_state.normalized_df.head())

st.markdown("---")
st.header("Step 5: Descriptive Statistics & Visualizations")
if st.button("Show Descriptive Statistics & Visualizations", key="btn_desc_stats"):
    if st.session_state.similarity_results is None:
        st.error("Please compute similarity scores first.")
    else:
        df = st.session_state.similarity_results.copy()
        st.subheader("Descriptive Statistics")
        st.write(df.describe())
        
        score_cols = [col for col in df.columns if "_" in col and col != "Text"]
        st.subheader("Histograms")
        for col in score_cols:
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, bins=30, ax=ax)
            ax.set_title(f"Histogram of {col}")
            st.pyplot(fig)
        
        st.subheader("Correlation Heatmap")
        corr = df[score_cols].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)

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
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)

st.markdown("---")
st.header("Step 7: Exploratory Factor Analysis (EFA)")
if st.button("Run Exploratory Factor Analysis (EFA)", key="btn_efa"):
    if st.session_state.similarity_results is None:
        st.error("Please compute similarity scores first.")
    else:
        # Monkey patch: make scipy.sum point to numpy.sum for compatibility
        import scipy
        scipy.sum = np.sum
        
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

def add_footer() -> None:
    """Add a persistent footer to all pages"""
    st.markdown("---")
    st.markdown("### **Gabriele Di Cicco, PhD in Social Psychology**")
    st.markdown("""
    [GitHub](https://github.com/gdc0000) | 
    [ORCID](https://orcid.org/0000-0002-1439-5790) | 
    [LinkedIn](https://www.linkedin.com/in/gabriele-di-cicco-124067b0/)
    """)
add.footer()
