from unittest.mock import MagicMock, patch

import torch

from app.ml import (
    _empty_embedding_tensor,
    _ensure_matching_rows,
    compute_similarity_scores_aggregated,
    compute_similarity_scores_item_by_item,
    compute_similarity_scores_single,
    generate_text_embeddings,
)


class TestEmbeddingHelpers:
    def test_empty_embedding_tensor(self):
        device = torch.device("cpu")
        tensor = _empty_embedding_tensor(device, 384)
        assert tensor.shape == (0, 384)
        assert tensor.device.type == "cpu"
        assert tensor.dtype == torch.float32

    def test_ensure_matching_rows_true(self):
        assert _ensure_matching_rows({"a": [1, 2], "b": [3, 4]}) is True

    def test_ensure_matching_rows_false(self):
        assert _ensure_matching_rows({"a": [1, 2], "b": [3]}) is False


class TestGenerateTextEmbeddings:
    @patch("app.ml._encode_in_batches")
    def test_calls_encode(self, mock_encode):
        mock_encode.return_value = torch.empty((2, 384))
        model = MagicMock()
        texts = ["hello", "world"]
        result = generate_text_embeddings(model, texts)
        mock_encode.assert_called_once_with(model, texts, show_progress=True)
        assert result.shape == (2, 384)


class TestComputeSimilarityAggregated:
    @patch("app.ml._get_text_series")
    @patch("app.ml._encode_in_batches")
    def test_aggregated_scoring(self, mock_encode, mock_get_text):
        mock_get_text.return_value.tolist.return_value = ["text1", "text2"]
        emb = torch.randn(2, 4)
        mock_encode.return_value = emb
        model = MagicMock()
        embeddings = torch.randn(2, 4)
        scales_data = {"ScaleA": ["item1", "item2"]}
        reverse_items = {}

        result = compute_similarity_scores_aggregated(model, embeddings, scales_data, reverse_items)
        assert result is not None
        assert list(result.columns) == ["Text", "ScaleA"]
        assert len(result) == 2

    @patch("app.ml._get_text_series")
    def test_empty_scales(self, mock_get_text):
        mock_get_text.return_value.tolist.return_value = ["text1"]
        model = MagicMock()
        embeddings = torch.randn(1, 384)
        result = compute_similarity_scores_aggregated(model, embeddings, {}, {})
        assert result is not None
        assert list(result.columns) == ["Text"]
        assert len(result) == 1


class TestComputeSimilarityItemByItem:
    @patch("app.ml._get_text_series")
    @patch("app.ml._encode_in_batches")
    def test_item_by_item_scoring(self, mock_encode, mock_get_text):
        mock_get_text.return_value.tolist.return_value = ["text1", "text2"]
        emb = torch.randn(2, 4)
        mock_encode.return_value = emb
        model = MagicMock()
        embeddings = torch.randn(2, 4)
        scales_data = {"ScaleA": ["item1", "item2"]}
        reverse_items = {}

        result = compute_similarity_scores_item_by_item(
            model, embeddings, scales_data, reverse_items
        )
        assert result is not None
        assert all(col.startswith("ScaleA_") for col in result.columns if col != "Text")
        assert len(result) == 2


class TestComputeSimilaritySingle:
    @patch("app.ml._get_text_series")
    @patch("app.ml._encode_in_batches")
    def test_single_construct_scoring(self, mock_encode, mock_get_text):
        mock_get_text.return_value.tolist.return_value = ["text1"]
        mock_encode.return_value = torch.randn(1, 4)
        model = MagicMock()
        embeddings = torch.randn(1, 4)
        constructs = [{"name": "C1", "text": "construct text"}]

        result = compute_similarity_scores_single(model, embeddings, constructs)
        assert result is not None
        assert "C1" in result.columns
        assert len(result) == 1
