import streamlit as st
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
from factor_analyzer import FactorAnalyzer
import time

# Set page configuration
st.set_page_config(page_title="BERT-based Text Analysis", layout="wide")

# Initialize session state variables
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

# Sidebar: Upload files and select model
st.sidebar.header("Input Files and Model Selection")

# Upload textual data file (CSV or Excel)
text_file = st.sidebar.file_uploader("Upload Text Data (CSV or Excel)", type=["csv", "xlsx"])
if text_file:
    try:
        if text_file.name.endswith("csv"):
            df_text = pd.read_csv(text_file)
        else:
            df_text = pd.read_excel(text_file)
        st.session_state.text_data = df_text
        st.sidebar.write("Preview of Text Data:")
        st.sidebar.dataframe(df_text.head())
        # Allow user to choose which column to analyze
        cols = df_text.columns.tolist()
        selected_column = st.sidebar.selectbox("Select the textual column", cols)
        st.session_state.text_column = selected_column
    except Exception as e:
        st.sidebar.error(f"Error reading text file: {e}")

# Upload validated scales file (Excel with multiple sheets)
scales_file = st.sidebar.file_uploader("Upload Validated Scales (Excel with sheets named after constructs)", type=["xlsx"])
if scales_file:
    try:
        xls = pd.ExcelFile(scales_file)
        scales_dict = {}
        for sheet_name in xls.sheet_names:
            df_sheet = pd.read_excel(xls, sheet_name=sheet_name)
            # Expecting a column named "Item" with the scale items
            if "Item" in df_sheet.columns:
                items = df_sheet["Item"].dropna().tolist()
                scales_dict[sheet_name] = items
        st.session_state.scales_data = scales_dict
        st.sidebar.write("Uploaded Scales:")
        st.sidebar.write(scales_dict)
        # For each scale, ask for reverse-scored item indices (0-indexed, comma separated)
        st.sidebar.subheader("Reverse Scored Items")
        reverse_dict = {}
        for scale in scales_dict.keys():
            rev = st.sidebar.text_input(f"Reverse items for {scale} (e.g., 1,3)", key=scale)
            if rev:
                try:
                    indices = [int(x.strip()) for x in rev.split(",") if x.strip().isdigit()]
                    reverse_dict[scale] = indices
                except Exception as e:
                    st.sidebar.error(f"Error for {scale}: {e}")
        st.session_state.reverse_items = reverse_dict
    except Exception as e:
        st.sidebar.error(f"Error reading scales file: {e}")

# Model selection: Provide a dropdown with descriptions
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
    st.success("Model loaded.")

# Main app title
st.title("BERT-based Text Analysis Application")

# Step 1: Embed textual data
if st.button("Step 1: Generate Text Embeddings"):
    if st.session_state.text_data is None or st.session_state.text_column is None:
        st.error("Please upload your text data and select the textual column.")
    else:
        texts = st.session_state.text_data[st.session_state.text_column].dropna().tolist()
        embeddings = []
        progress_bar = st.progress(0)
        st.info("Generating embeddings for textual data...")
        for i, text in enumerate(texts):
            embedding = st.session_state.model_instance.encode(text, convert_to_tensor=True)
            embeddings.append(embedding)
            progress_bar.progress((i + 1) / len(texts))
            time.sleep(0.01)  # simulate delay
        st.session_state.text_embeddings = torch.stack(embeddings)
        st.success("Text embeddings generated successfully.")

# Step 2: Embed scales and compute similarity scores
if st.button("Step 2: Compute Similarity Scores for Scales"):
    # Use the same filtered texts used for embeddings
    texts = st.session_state.text_data[st.session_state.text_column].dropna().tolist()
    
    if st.session_state.scales_data == {}:
        st.error("Please upload the validated scales file first.")
    elif st.session_state.text_embeddings is None:
        st.error("Please generate text embeddings first.")
    else:
        results = {"Text": texts}
        for scale, items in st.session_state.scales_data.items():
            st.write(f"Processing scale: {scale}")
            # Compute embeddings for each scale item
            item_embeds = st.session_state.model_instance.encode(items, convert_to_tensor=True)
            sims = util.cos_sim(st.session_state.text_embeddings, item_embeds)
            # Apply reverse scoring if specified for this scale
            if scale in st.session_state.reverse_items:
                rev_idx = st.session_state.reverse_items[scale]
                sims[:, rev_idx] = 1 - sims[:, rev_idx]
            # Aggregate by averaging across the items
            agg_scores = sims.cpu().numpy().mean(axis=1)
            results[f"{scale}_score"] = agg_scores
        
        # Debug: Print lengths of arrays to ensure consistency
        lengths = {key: len(val) for key, val in results.items()}
        st.write("Array lengths in results:", lengths)
        if len(set(lengths.values())) > 1:
            st.error("Mismatch in data lengths. Please ensure that your text column has no missing values.")
        else:
            st.session_state.similarity_results = pd.DataFrame(results)
            st.success("Similarity scores computed and compiled.")
            st.write(st.session_state.similarity_results.head())


# Step 3: Exclude outliers based on Z-scores
if st.button("Step 3: Exclude Outliers (Z-score > 3)"):
    if st.session_state.similarity_results is None:
        st.error("Please compute similarity scores first.")
    else:
        df = st.session_state.similarity_results.copy()
        score_cols = [col for col in df.columns if col.endswith("_score")]
        z_scores = df[score_cols].apply(zscore)
        mask = (np.abs(z_scores) <= 3).all(axis=1)
        df_no_outliers = df[mask]
        st.session_state.similarity_results = df_no_outliers
        st.success("Outliers excluded.")
        st.write(f"Data shape after outlier removal: {df_no_outliers.shape}")

# Step 4: Normalize the data (Min-Max Normalization)
if st.button("Step 4: Normalize Data"):
    if st.session_state.similarity_results is None:
        st.error("Please compute similarity scores first.")
    else:
        df = st.session_state.similarity_results.copy()
        score_cols = [col for col in df.columns if col.endswith("_score")]
        df_norm = df.copy()
        for col in score_cols:
            df_norm[col] = (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min())
        st.session_state.normalized_df = df_norm
        st.success("Data normalized successfully.")
        st.write(st.session_state.normalized_df.head())

# Step 5: Descriptive statistics and visualization
if st.button("Step 5: Show Descriptive Statistics & Visualizations"):
    if st.session_state.similarity_results is None:
        st.error("Please compute similarity scores first.")
    else:
        df = st.session_state.similarity_results.copy()
        st.subheader("Descriptive Statistics")
        st.write(df.describe())

        score_cols = [col for col in df.columns if col.endswith("_score")]
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

# Step 6: Correlation Analysis
if st.button("Step 6: Run Correlation Analysis"):
    if st.session_state.similarity_results is None:
        st.error("Please compute similarity scores first.")
    else:
        df = st.session_state.similarity_results.copy()
        score_cols = [col for col in df.columns if col.endswith("_score")]
        corr_matrix = df[score_cols].corr()
        st.subheader("Correlation Matrix")
        st.dataframe(corr_matrix)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)

# Step 7: Exploratory Factor Analysis (EFA)
if st.button("Step 7: Run Exploratory Factor Analysis (EFA)"):
    if st.session_state.similarity_results is None:
        st.error("Please compute similarity scores first.")
    else:
        df = st.session_state.similarity_results.copy()
        score_cols = [col for col in df.columns if col.endswith("_score")]
        data_for_efa = df[score_cols]
        fa = FactorAnalyzer(n_factors=2, rotation="varimax")
        fa.fit(data_for_efa)
        eigenvalues, _ = fa.get_eigenvalues()
        loadings = pd.DataFrame(fa.loadings_, index=score_cols)
        st.subheader("Eigenvalues")
        st.write(eigenvalues)
        st.subheader("Factor Loadings")
        st.write(loadings)

st.write("Session State Keys:", list(st.session_state.keys()))
