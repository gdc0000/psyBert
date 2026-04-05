import pandas as pd
import streamlit as st

from app.analysis import compute_corr_with_significance_optimized
from app.config import MODEL_OPTIONS, SCORING_OPTIONS
from app.data import load_scales, load_text_data
from app.ml import (
    compute_similarity_scores_aggregated,
    compute_similarity_scores_item_by_item,
    compute_similarity_scores_single,
    generate_text_embeddings,
)


def render_sidebar() -> None:
    st.sidebar.title("Configuration")

    with st.sidebar.expander("Data Input", expanded=True):
        text_file = st.file_uploader(
            "Upload Text Data (CSV or Excel)", type=["csv", "xlsx"], key="text_file"
        )
        if text_file:
            file_type = "csv" if text_file.name.endswith("csv") else "xlsx"
            try:
                st.session_state.text_data = load_text_data(text_file, file_type)
                st.write("**Text Data Preview:**")
                st.dataframe(st.session_state.text_data.head())
                cols = st.session_state.text_data.columns.tolist()
                st.session_state.text_column = st.selectbox("Select Textual Column", cols)
            except Exception as exc:
                st.error(f"Error loading text file: {exc}")

    with st.sidebar.expander("Scoring Method & Constructs", expanded=True):
        scoring_method = st.radio(
            "Select Scoring Method:",
            options=SCORING_OPTIONS,
            key="scoring_method",
        )
        st.session_state.method = scoring_method

        if scoring_method in [
            "Aggregated Items (Excel Upload)",
            "Item-by-item (Excel Upload)",
        ]:
            scales_file = st.file_uploader("Upload Scales (Excel)", type=["xlsx"], key="scales_file")
            if scales_file:
                try:
                    all_scales_data, all_reverse_items_dict = load_scales(scales_file)
                    available_scales = list(all_scales_data.keys())
                    selected_scales = st.multiselect(
                        "Select Scales",
                        options=available_scales,
                        default=available_scales,
                        key="selected_scales",
                    )
                    selected_scales_data, selected_reverse_items = {}, {}
                    for scale in selected_scales:
                        items = all_scales_data[scale]
                        st.write(f"**Review Reverse Items for {scale}:**")
                        df_preview = pd.DataFrame({"Index": list(range(len(items))), "Item": items})
                        st.dataframe(df_preview)
                        user_rev = st.multiselect(
                            f"Select reverse indices for {scale}",
                            options=list(range(len(items))),
                            default=all_reverse_items_dict[scale],
                            key=f"rev_{scale}",
                        )
                        selected_scales_data[scale] = items
                        selected_reverse_items[scale] = user_rev
                    st.session_state.scales_data = selected_scales_data
                    st.session_state.reverse_items = selected_reverse_items
                except Exception as exc:
                    st.error(f"Error loading scales file: {exc}")
        else:
            st.subheader("Add Construct")
            construct_name = st.text_input("Construct Name", key="construct_name")
            construct_text = st.text_area("Construct Text", key="construct_text")
            if st.button("Add Construct", key="btn_add_construct"):
                if construct_name and construct_text:
                    st.session_state.constructs.append(
                        {"name": construct_name, "text": construct_text}
                    )
                    st.success(f"Construct '{construct_name}' added.")
                else:
                    st.error("Provide both name and text for the construct.")
            if st.session_state.constructs:
                st.write("**Current Constructs:**")
                for construct in st.session_state.constructs:
                    st.write(f"- {construct['name']}")

    with st.sidebar.expander("Model Selection", expanded=True):
        selected_model = st.selectbox("Choose Model", list(MODEL_OPTIONS.keys()), index=0)
        st.write(MODEL_OPTIONS[selected_model])
        st.session_state.selected_model = selected_model


def render_tabs() -> None:
    tabs = st.tabs(["Overview", "Embeddings", "Similarity", "Analysis", "Download"])

    with tabs[0]:
        st.header("Welcome to the BERT-based Text Analysis App")
        st.markdown(
            """
        This application allows you to analyze text data using advanced sentence embeddings.

        **Workflow Overview:**
        - **Data Input:** Upload your text data and (if applicable) validated scales or define constructs.
        - **Embeddings:** Generate embeddings for your text.
        - **Similarity:** Compute similarity scores between your text and scale items or constructs.
        - **Analysis:** View descriptive statistics and a correlation matrix (optimized for speed).
        - **Download:** Export your computed similarity score table.
        """
        )
        st.info(
            "Use the sidebar to configure your data, scoring method, and model. Then navigate through the tabs."
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
                texts = (
                    st.session_state.text_data[st.session_state.text_column]
                    .dropna()
                    .tolist()
                )
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


def add_footer() -> None:
    st.markdown("---")
    st.markdown("### **Gabriele Di Cicco, PhD in Social Psychology**")
    st.markdown(
        "[GitHub](https://github.com/gdc0000) | "
        "[ORCID](https://orcid.org/0000-0002-1439-5790) | "
        "[LinkedIn](https://www.linkedin.com/in/gabriele-di-cicco-124067b0/)"
    )

