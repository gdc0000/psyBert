from unittest.mock import MagicMock, patch

import pandas as pd

from app.data import load_scales, load_text_data


class TestLoadTextData:
    @patch("app.data.pd.read_csv")
    def test_csv_file_type(self, mock_read_csv):
        mock_file = MagicMock()
        mock_read_csv.return_value = pd.DataFrame({"col1": [1, 2]})
        result = load_text_data(mock_file, "csv")
        mock_read_csv.assert_called_once_with(mock_file)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["col1"]

    @patch("app.data.pd.read_excel")
    def test_excel_file_type(self, mock_read_excel):
        mock_file = MagicMock()
        mock_read_excel.return_value = pd.DataFrame({"col1": [3, 4]})
        result = load_text_data(mock_file, "xlsx")
        mock_read_excel.assert_called_once_with(mock_file)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["col1"]


class TestLoadScales:
    @patch("app.data.pd.ExcelFile")
    def test_loads_all_sheets(self, mock_excel_file):
        mock_instance = MagicMock()
        mock_instance.sheet_names = ["Scale1", "Scale2"]
        mock_excel_file.return_value = mock_instance

        df1 = pd.DataFrame({"Item": ["item1", "item2"], "Rev": [0.0, 1.0]})
        df2 = pd.DataFrame({"Item": ["item3"], "Rev": [0.0]})
        mock_read = MagicMock(side_effect=[df1, df2])

        with patch("app.data.pd.read_excel", mock_read):
            scales_data, reverse_items = load_scales(MagicMock())

        assert "Scale1" in scales_data
        assert "Scale2" in scales_data
        assert scales_data["Scale1"] == ["item1", "item2"]
        assert reverse_items["Scale1"] == [1]

    @patch("app.data.pd.ExcelFile")
    def test_skips_sheets_without_item_and_rev(self, mock_excel_file):
        mock_instance = MagicMock()
        mock_instance.sheet_names = ["BadSheet"]
        mock_excel_file.return_value = mock_instance

        df_bad = pd.DataFrame({"Foo": [1], "Bar": [2]})
        mock_read = MagicMock(return_value=df_bad)

        with patch("app.data.pd.read_excel", mock_read):
            scales_data, reverse_items = load_scales(MagicMock())

        assert scales_data == {}
        assert reverse_items == {}

    @patch("app.data.pd.ExcelFile")
    def test_handles_empty_sheets(self, mock_excel_file):
        mock_instance = MagicMock()
        mock_instance.sheet_names = ["Empty"]
        mock_excel_file.return_value = mock_instance

        df_empty = pd.DataFrame({"Item": [None, None], "Rev": [None, None]})
        mock_read = MagicMock(return_value=df_empty)

        with patch("app.data.pd.read_excel", mock_read):
            scales_data, reverse_items = load_scales(MagicMock())

        assert scales_data["Empty"] == []
        assert reverse_items["Empty"] == []
