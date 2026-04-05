import sys

# Patch for Python 3.12: if 'distutils.dir_util' is missing, inject it from setuptools.
try:
    import distutils.dir_util
except ModuleNotFoundError:
    from setuptools._distutils import dir_util as distutils_dir_util

    sys.modules["distutils.dir_util"] = distutils_dir_util

import logging

import streamlit as st

from app.config import configure_page
from app.ml import get_model
from app.state import init_session_state
from app.ui import add_footer, render_sidebar, render_tabs


logging.basicConfig(level=logging.INFO)


def run_app() -> None:
    configure_page()
    init_session_state()
    render_sidebar()

    if st.session_state.get("model_instance") is None:
        with st.spinner("Loading embedding model..."):
            st.session_state.model_instance = get_model(st.session_state.selected_model)
        st.success("Model loaded.")

    render_tabs()
    add_footer()


if __name__ == "__main__":
    run_app()
