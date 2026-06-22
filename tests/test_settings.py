from app.settings import EMBED_BATCH_SIZE, MODEL_OPTIONS, PAGE_TITLE, SCORING_OPTIONS


class TestSettings:
    def test_model_options_is_dict(self):
        assert isinstance(MODEL_OPTIONS, dict)
        assert len(MODEL_OPTIONS) >= 3

    def test_scoring_options_is_list(self):
        assert isinstance(SCORING_OPTIONS, list)
        assert len(SCORING_OPTIONS) == 3

    def test_page_title_is_string(self):
        assert isinstance(PAGE_TITLE, str)
        assert len(PAGE_TITLE) > 0

    def test_embed_batch_size_is_positive_int(self):
        assert isinstance(EMBED_BATCH_SIZE, int)
        assert EMBED_BATCH_SIZE > 0
