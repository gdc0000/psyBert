from unittest.mock import MagicMock, patch

from app.state import init_session_state


class TestInitSessionState:
    @patch("app.state.st.session_state", new_callable=MagicMock)
    def test_sets_defaults_when_keys_missing(self, mock_session_state):
        mock_session_state.__contains__.return_value = False
        init_session_state()
        expected_keys = [
            "constructs",
            "similarity_results",
            "model_instance",
            "method",
            "text_data",
            "text_column",
            "text_embeddings",
            "scales_data",
            "reverse_items",
            "selected_model",
        ]
        for key in expected_keys:
            assert mock_session_state.__setitem__.called
            call_args = [c for c in mock_session_state.__setitem__.call_args_list if c[0][0] == key]
            assert len(call_args) > 0, f"Key '{key}' was not set"

    @patch("app.state.st.session_state", new_callable=MagicMock)
    def test_does_not_override_existing_keys(self, mock_session_state):
        mock_session_state.__contains__.return_value = True
        init_session_state()
        mock_session_state.__setitem__.assert_not_called()
