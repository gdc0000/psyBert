import streamlit as st

from app.settings import PAGE_TITLE


def configure_page() -> None:
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
