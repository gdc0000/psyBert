import time

import numpy as np
import pandas as pd
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer, util
from typing import Optional


@st.cache_resource(show_spinner=False)
def get_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def generate_text_embeddings(model: SentenceTransformer, texts: list) -> torch.Tensor:
    embeddings = []
    progress_bar = st.progress(0)
    total = len(texts)
    for i, text in enumerate(texts):
        embeddings.append(model.encode(text, convert_to_tensor=True))
        progress_bar.progress((i + 1) / total)
        time.sleep(0.01)
    return torch.stack(embeddings)


def compute_similarity_scores_aggregated(
    model: SentenceTransformer,
    text_embeddings: torch.Tensor,
    scales_data: dict,
    reverse_items: dict,
) -> Optional[pd.DataFrame]:
    texts = st.session_state.text_data[st.session_state.text_column].dropna().tolist()
    results = {"Text": texts}
    for scale, items in scales_data.items():
        item_scores = []
        for i, item in enumerate(items):
            item_embed = model.encode(item, convert_to_tensor=True)
            sims = util.cos_sim(text_embeddings, item_embed.unsqueeze(0))
            sims_np = sims.cpu().numpy().flatten()
            if i in reverse_items.get(scale, []):
                sims_np = 1 - sims_np
            item_scores.append(sims_np)
        results[scale] = np.mean(np.array(item_scores), axis=0)
    if len({len(v) for v in results.values()}) != 1:
        st.error("Mismatch in data lengths. Please check for missing values.")
        return None
    return pd.DataFrame(results)


def compute_similarity_scores_item_by_item(
    model: SentenceTransformer,
    text_embeddings: torch.Tensor,
    scales_data: dict,
    reverse_items: dict,
) -> Optional[pd.DataFrame]:
    texts = st.session_state.text_data[st.session_state.text_column].dropna().tolist()
    results = {"Text": texts}
    for scale, items in scales_data.items():
        for i, item in enumerate(items):
            item_embed = model.encode(item, convert_to_tensor=True)
            sims = util.cos_sim(text_embeddings, item_embed.unsqueeze(0))
            sims_np = sims.cpu().numpy().flatten()
            if i in reverse_items.get(scale, []):
                sims_np = 1 - sims_np
            results[f"{scale}_{i + 1}"] = sims_np
    if len({len(v) for v in results.values()}) != 1:
        st.error("Mismatch in data lengths. Please check for missing values.")
        return None
    return pd.DataFrame(results)


def compute_similarity_scores_single(
    model: SentenceTransformer, text_embeddings: torch.Tensor, constructs: list
) -> Optional[pd.DataFrame]:
    texts = st.session_state.text_data[st.session_state.text_column].dropna().tolist()
    results = {"Text": texts}
    for construct in constructs:
        name = construct["name"]
        construct_embed = model.encode(construct["text"], convert_to_tensor=True).unsqueeze(0)
        sims = util.cos_sim(text_embeddings, construct_embed)
        results[name] = sims.cpu().numpy().flatten()
    if len({len(v) for v in results.values()}) != 1:
        st.error("Mismatch in data lengths. Please check for missing values.")
        return None
    return pd.DataFrame(results)
