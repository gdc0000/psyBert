import streamlit as st


def init_session_state() -> None:
    defaults = {
        "constructs": [],
        "similarity_results": None,
        "model_instance": None,
        "method": None,
        "text_data": None,
        "text_column": None,
        "text_embeddings": None,
        "scales_data": None,
        "reverse_items": None,
        "selected_model": "all-MiniLM-L6-v2",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
