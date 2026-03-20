"""Tests for the RegexMapper percentage extraction logic."""

import pytest

from topreward.mapper.regex_mapper import RegexMapper


@pytest.fixture
def mapper():
    return RegexMapper()


def test_single_percentage_per_frame(mapper):
    """Each frame label followed by a percentage is extracted in order."""
    response = "Frame 1: 10%, Frame 2: 50%, Frame 3: 90%"
    result = mapper.extract_percentages(response)
    assert result == [10, 50, 90]


def test_no_percentages_returns_empty(mapper):
    """A response with no percent signs produces an empty list."""
    response = "There are no percentages here at all."
    result = mapper.extract_percentages(response)
    assert result == []


def test_plain_numbers_without_percent_sign(mapper):
    """Numbers not followed by '%' are ignored by the regex extractor."""
    response = "Frame 1: 50 Frame 2: 75"
    result = mapper.extract_percentages(response)
    assert result == []


def test_values_are_rounded_integers(mapper):
    """Fractional percentages are rounded with largest remainder and sum to 100."""
    response = "Frame 1: 33.3%, Frame 2: 33.3%, Frame 3: 33.4%"
    result = mapper.extract_percentages(response)
    assert isinstance(result, list)
    assert all(isinstance(v, (int, float)) for v in result)
    assert len(result) == 3
    assert all(v == round(v) for v in result)
    assert sum(result) == 100


def test_multiple_values_in_one_line(mapper):
    """Multiple percent values on one line are all extracted."""
    response = "10%, 20%, 30%, 40%"
    result = mapper.extract_percentages(response)
    assert len(result) == 4


def test_out_of_range_values_excluded(mapper):
    """Values outside [0, 100] are silently dropped."""
    response = "Frame 1: 150%, Frame 2: 50%, Frame 3: -5%"
    result = mapper.extract_percentages(response)
    assert result == [50]


def test_zero_and_hundred_included(mapper):
    """Boundary values 0% and 100% are valid and included."""
    response = "Start: 0%, End: 100%"
    result = mapper.extract_percentages(response)
    assert result == [0, 100]


def test_integer_values_not_scaled(mapper):
    """Purely integer percentages are returned as-is, not normalized to 100."""
    response = "Frame 1: 10%, Frame 2: 30%"
    result = mapper.extract_percentages(response)
    assert result == [10, 30]
