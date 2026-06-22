import pandas as pd
import streamlit as st

from app.data import load_scales, load_text_data
from app.settings import MODEL_OPTIONS, SCORING_OPTIONS


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
            scales_file = st.file_uploader(
                "Upload Scales (Excel)", type=["xlsx"], key="scales_file"
            )
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
