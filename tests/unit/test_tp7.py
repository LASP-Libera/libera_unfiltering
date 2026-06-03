"""Unit tests for tp7/tp7.py — pure logic, no file I/O."""
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tp7.tp7 import Tape7, _as_path


class TestGetTitle:
    @pytest.mark.parametrize("filename,expected", [
        ("lnd_sza00_something.tp7", "Land"),
        ("ocecld_sza00.tp7", "Cloudy Ocean"),
        ("oceclr_sza41.tp7", "Clear Ocean"),
        ("sno_sza60.tp7", "Snow"),
        ("dc_sza75.tp7", "Deep Convective Cloud"),
    ])
    def test_known_prefixes(self, filename, expected):
        p = MagicMock()
        p.name = filename
        assert Tape7._get_title(p) == expected

    def test_unknown_prefix_raises(self):
        p = MagicMock()
        p.name = "unknown_scene_file.tp7"
        with pytest.raises(ValueError, match="Unrecognised scene prefix"):
            Tape7._get_title(p)

    def test_empty_name_raises(self):
        p = MagicMock()
        p.name = "_noprefix.tp7"
        with pytest.raises((ValueError, KeyError)):
            Tape7._get_title(p)


class TestParseMetadata:
    def _make_lines(self, first_char, num_cols, num_runs=1):
        """Build a minimal fake tp7 line list suitable for _parse_metadata."""
        data_row = " ".join(["1.23456"] * num_cols) + "\n"
        terminator = "ENDMARKER\n"
        lines = [f"{first_char}some header content\n"]
        for _ in range(num_runs):
            lines.extend(["header_line\n"] * 3)
            lines.append(data_row)   # lines[-2] when terminator is last
            lines.append(terminator) # lines[-1]
        return lines

    def test_daytime_returns_size_4000(self):
        lines = self._make_lines('F', 12)
        size, _, _, _ = Tape7._parse_metadata(lines)
        assert size == 4000

    def test_nighttime_returns_size_4996(self):
        lines = self._make_lines('N', 12)
        size, _, _, _ = Tape7._parse_metadata(lines)
        assert size == 4996

    def test_12col_num_cols_and_start_index(self):
        lines = self._make_lines('F', 12)
        _, num_cols, _, start_index = Tape7._parse_metadata(lines)
        assert num_cols == 12
        assert start_index == 13

    def test_14col_num_cols_and_start_index(self):
        lines = self._make_lines('F', 14)
        _, num_cols, _, start_index = Tape7._parse_metadata(lines)
        assert num_cols == 14
        assert start_index == 12

    def test_num_runs_counted_correctly(self):
        lines = self._make_lines('F', 12, num_runs=3)
        _, _, num_runs, _ = Tape7._parse_metadata(lines)
        assert num_runs == 3


class TestAsPath:
    def test_string_input_has_open_and_name(self):
        p = _as_path("/some/path/file.tp7")
        assert hasattr(p, 'open')
        assert hasattr(p, 'name')

    def test_path_object_is_returned_unchanged(self):
        p = Path("/some/path/file.tp7")
        result = _as_path(p)
        assert result is p

    def test_path_object_passthrough_preserves_type(self):
        p = Path("/tmp/file.tp7")
        assert isinstance(_as_path(p), Path)
