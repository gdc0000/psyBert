import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from app.analysis import compute_corr_with_significance_optimized
from app.data import load_scales, load_text_data
from app.ml import (
    compute_similarity_scores_aggregated,
    compute_similarity_scores_item_by_item,
    compute_similarity_scores_single,
    generate_text_embeddings,
)


def load_text(file, file_type: str) -> pd.DataFrame:
    return load_text_data(file, file_type)


def load_scales_file(file) -> tuple[dict, dict]:
    return load_scales(file)


def generate_embeddings(model: SentenceTransformer, texts: list[str]) -> torch.Tensor:
    return generate_text_embeddings(model, texts)


def compute_similarity(
    method: str,
    model: SentenceTransformer,
    text_embeddings: torch.Tensor,
    texts: list[str],
    scales_data: dict | None = None,
    reverse_items: dict | None = None,
    constructs: list[dict] | None = None,
) -> pd.DataFrame | None:
    if method in ("Aggregated Items (Excel Upload)",):
        return compute_similarity_scores_aggregated(
            model, text_embeddings, scales_data or {}, reverse_items or {}
        )
    if method in ("Item-by-item (Excel Upload)",):
        return compute_similarity_scores_item_by_item(
            model, text_embeddings, scales_data or {}, reverse_items or {}
        )
    return compute_similarity_scores_single(model, text_embeddings, constructs or [])


def compute_correlation(df: pd.DataFrame) -> pd.DataFrame:
    return compute_corr_with_significance_optimized(df)


def describe_results(df: pd.DataFrame) -> pd.DataFrame:
    return df.describe()


def results_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")
