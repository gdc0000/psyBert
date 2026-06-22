import pandas as pd
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer, util

from app.settings import EMBED_BATCH_SIZE


@st.cache_resource(show_spinner=False)
def get_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def _get_text_series() -> pd.Series:
    return st.session_state.text_data[st.session_state.text_column].dropna()


def _empty_embedding_tensor(device: torch.device, emb_dim: int) -> torch.Tensor:
    return torch.empty((0, emb_dim), dtype=torch.float32, device=device)


def _encode_in_batches(
    model: SentenceTransformer, texts: list, show_progress: bool = False
) -> torch.Tensor:
    if not texts:
        model_device = model.device if hasattr(model, "device") else torch.device("cpu")
        emb_dim = model.get_sentence_embedding_dimension() or 768
        return _empty_embedding_tensor(model_device, emb_dim)

    chunks = []
    progress = st.progress(0) if show_progress else None
    total = len(texts)

    with torch.inference_mode():
        for start in range(0, total, EMBED_BATCH_SIZE):
            batch = texts[start : start + EMBED_BATCH_SIZE]
            encoded = model.encode(
                batch,
                convert_to_tensor=True,
                show_progress_bar=False,
                batch_size=EMBED_BATCH_SIZE,
            )
            if encoded.ndim == 1:
                encoded = encoded.unsqueeze(0)
            chunks.append(encoded)
            if progress is not None:
                progress.progress(min((start + len(batch)) / total, 1.0))

    if progress is not None:
        progress.empty()

    return torch.cat(chunks, dim=0)


def _ensure_matching_rows(results: dict) -> bool:
    return len({len(v) for v in results.values()}) == 1


def _finalize_results(results: dict) -> pd.DataFrame | None:
    if not _ensure_matching_rows(results):
        st.error("Mismatch in data lengths. Please check for missing values.")
        return None
    return pd.DataFrame(results)


def generate_text_embeddings(model: SentenceTransformer, texts: list) -> torch.Tensor:
    return _encode_in_batches(model, texts, show_progress=True)


def compute_similarity_scores_aggregated(
    model: SentenceTransformer,
    text_embeddings: torch.Tensor,
    scales_data: dict,
    reverse_items: dict,
) -> pd.DataFrame | None:
    texts = _get_text_series().tolist()
    results = {"Text": texts}
    text_embeddings = text_embeddings.float()

    for scale, items in scales_data.items():
        if not items:
            results[scale] = [float("nan")] * len(texts)
            continue

        with torch.inference_mode():
            item_embeddings = _encode_in_batches(model, items)
            sims = util.cos_sim(text_embeddings, item_embeddings)
            rev_idx = reverse_items.get(scale, [])
            if rev_idx:
                valid_rev_idx = [idx for idx in rev_idx if 0 <= idx < sims.shape[1]]
                if valid_rev_idx:
                    sims = sims.clone()
                    sims[:, valid_rev_idx] = 1 - sims[:, valid_rev_idx]
            results[scale] = sims.mean(dim=1).cpu().numpy()  # type: ignore[assignment]

    return _finalize_results(results)


def compute_similarity_scores_item_by_item(
    model: SentenceTransformer,
    text_embeddings: torch.Tensor,
    scales_data: dict,
    reverse_items: dict,
) -> pd.DataFrame | None:
    texts = _get_text_series().tolist()
    results = {"Text": texts}
    text_embeddings = text_embeddings.float()

    for scale, items in scales_data.items():
        if not items:
            continue

        with torch.inference_mode():
            item_embeddings = _encode_in_batches(model, items)
            sims = util.cos_sim(text_embeddings, item_embeddings)
            rev_idx = reverse_items.get(scale, [])
            if rev_idx:
                valid_rev_idx = [idx for idx in rev_idx if 0 <= idx < sims.shape[1]]
                if valid_rev_idx:
                    sims = sims.clone()
                    sims[:, valid_rev_idx] = 1 - sims[:, valid_rev_idx]
            sims_np = sims.cpu().numpy()
        for i in range(sims_np.shape[1]):
            results[f"{scale}_{i + 1}"] = sims_np[:, i]  # type: ignore[assignment]

    return _finalize_results(results)


def compute_similarity_scores_single(
    model: SentenceTransformer, text_embeddings: torch.Tensor, constructs: list
) -> pd.DataFrame | None:
    texts = _get_text_series().tolist()
    results = {"Text": texts}

    names = [construct["name"] for construct in constructs]
    construct_texts = [construct["text"] for construct in constructs]
    if not construct_texts:
        return _finalize_results(results)

    with torch.inference_mode():
        text_embeddings = text_embeddings.float()
        construct_embeddings = _encode_in_batches(model, construct_texts)
        sims = util.cos_sim(text_embeddings, construct_embeddings).cpu().numpy()

    for i, name in enumerate(names):
        results[name] = sims[:, i]  # type: ignore[assignment]

    return _finalize_results(results)
