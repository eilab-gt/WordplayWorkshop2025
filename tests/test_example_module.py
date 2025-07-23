"""Tests for example module."""

import pytest

from src.example import add, multiply


class TestExample:
    """Test cases for example module functions."""

    def test_add(self):
        """Test addition function."""
        assert add(2, 3) == 5
        assert add(-1, 1) == 0
        assert add(0, 0) == 0

    def test_multiply(self):
        """Test multiplication function."""
        assert multiply(2, 3) == 6
        assert multiply(-2, 3) == -6
        assert multiply(0, 100) == 0

    @pytest.mark.parametrize(
        "a,b,expected",
        [
            (1, 1, 1),
            (2, 3, 6),
            (0, 5, 0),
            (-2, -3, 6),
        ],
    )
    def test_multiply_parametrized(self, a: int, b: int, expected: int):
        """Test multiplication with multiple inputs."""
        assert multiply(a, b) == expected
