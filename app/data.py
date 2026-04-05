import pandas as pd
import streamlit as st
from typing import Dict, Tuple


@st.cache_data(show_spinner=False)
def load_text_data(file, file_type: str) -> pd.DataFrame:
    if file_type == "csv":
        return pd.read_csv(file)
    return pd.read_excel(file)


@st.cache_data(show_spinner=False)
def load_scales(file) -> Tuple[Dict, Dict]:
    xls = pd.ExcelFile(file)
    scales_data, reverse_items_dict = {}, {}
    for sheet_name in xls.sheet_names:
        df_sheet = pd.read_excel(xls, sheet_name=sheet_name)
        if {"Item", "Rev"}.issubset(df_sheet.columns):
            df_sheet = df_sheet.dropna(subset=["Item"])
            items = df_sheet["Item"].tolist()
            try:
                computed_rev = [
                    i
                    for i, val in enumerate(df_sheet["Rev"].tolist())
                    if float(val) == 1.0
                ]
            except Exception as exc:
                st.sidebar.error(
                    f"Error processing reverse items in sheet '{sheet_name}': {exc}"
                )
                computed_rev = []
            scales_data[sheet_name] = items
            reverse_items_dict[sheet_name] = computed_rev
        else:
            st.sidebar.error(
                f"Sheet '{sheet_name}' must have both 'Item' and 'Rev' columns."
            )
    return scales_data, reverse_items_dict
