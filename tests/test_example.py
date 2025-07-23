"""Example test file to verify pytest is working correctly."""


def test_example():
    """Simple test to verify pytest setup."""
    assert 1 + 1 == 2


def test_example_with_fixture():
    """Test using basic Python features."""
    data = [1, 2, 3, 4, 5]
    assert sum(data) == 15
    assert len(data) == 5
