import streamlit as st

from app.analysis import compute_corr_with_significance_optimized
from app.ml import (
    compute_similarity_scores_aggregated,
    compute_similarity_scores_item_by_item,
    compute_similarity_scores_single,
    generate_text_embeddings,
)


def render_tabs() -> None:
    tabs = st.tabs(["Overview", "Embeddings", "Similarity", "Analysis", "Download"])

    with tabs[0]:
        st.header("Welcome to the BERT-based Text Analysis App")
        st.markdown(
            """
        This application allows you to analyze text data using advanced sentence embeddings.

        **Workflow Overview:**
        - **Data Input:** Upload text data with validated scales or define constructs.
        - **Embeddings:** Generate embeddings for your text.
        - **Similarity:** Compute similarity scores between your text and scale items or constructs.
        - **Analysis:** View descriptive statistics and a correlation matrix (optimized for speed).
        - **Download:** Export your computed similarity score table.
        """
        )
        st.info(
            "Use the sidebar to configure data, scoring method, and model. "
            "Then navigate through the tabs."
        )

    with tabs[1]:
        st.header("Step 1: Generate Text Embeddings")
        if st.button("Generate Embeddings", key="btn_gen_text_embed"):
            if (
                st.session_state.get("text_data") is None
                or st.session_state.get("text_column") is None
            ):
                st.error("Upload text data and select the textual column from the sidebar.")
            else:
                texts = st.session_state.text_data[st.session_state.text_column].dropna().tolist()
                with st.spinner("Generating embeddings..."):
                    embeddings = generate_text_embeddings(st.session_state.model_instance, texts)
                st.session_state.text_embeddings = embeddings
                st.success("Embeddings generated!")
                st.write("Embeddings shape:", embeddings.shape)

    with tabs[2]:
        st.header("Step 2: Compute Similarity Scores")
        if st.button("Compute Similarity", key="btn_compute_sim"):
            if st.session_state.get("text_embeddings") is None:
                st.error("Please generate embeddings first (see 'Embeddings' tab).")
            else:
                method = st.session_state.method
                if method in [
                    "Aggregated Items (Excel Upload)",
                    "Item-by-item (Excel Upload)",
                ]:
                    if not st.session_state.get("scales_data"):
                        st.error("Upload validated scales in the sidebar.")
                        sim_df = None
                    else:
                        with st.spinner("Computing similarity scores..."):
                            if method == "Aggregated Items (Excel Upload)":
                                sim_df = compute_similarity_scores_aggregated(
                                    st.session_state.model_instance,
                                    st.session_state.text_embeddings,
                                    st.session_state.scales_data,
                                    st.session_state.reverse_items,
                                )
                            else:
                                sim_df = compute_similarity_scores_item_by_item(
                                    st.session_state.model_instance,
                                    st.session_state.text_embeddings,
                                    st.session_state.scales_data,
                                    st.session_state.reverse_items,
                                )
                else:
                    if not st.session_state.constructs:
                        st.error("Please add at least one construct in the sidebar.")
                        sim_df = None
                    else:
                        with st.spinner("Computing similarity scores..."):
                            sim_df = compute_similarity_scores_single(
                                st.session_state.model_instance,
                                st.session_state.text_embeddings,
                                st.session_state.constructs,
                            )
                if sim_df is not None:
                    st.session_state.similarity_results = sim_df
                    st.success("Similarity scores computed!")
                    st.dataframe(sim_df.head())

    with tabs[3]:
        st.header("Step 3: Analysis")
        if st.session_state.get("similarity_results") is None:
            st.error("Compute similarity scores first (see 'Similarity' tab).")
        else:
            st.subheader("Descriptive Statistics")
            st.dataframe(st.session_state.similarity_results.describe())
            st.subheader("Correlation Matrix with Significance")
            corr_annotated = compute_corr_with_significance_optimized(
                st.session_state.similarity_results
            )
            st.dataframe(corr_annotated)

    with tabs[4]:
        st.header("Step 4: Download Data")
        if st.session_state.get("similarity_results") is not None:
            csv_data = st.session_state.similarity_results.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Similarity Score Table (CSV)",
                data=csv_data,
                file_name="similarity_scores.csv",
                mime="text/csv",
            )
        else:
            st.error("Compute similarity scores to enable data download.")
