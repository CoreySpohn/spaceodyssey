import spaceodyssey


def test_version_is_a_string():
    assert isinstance(spaceodyssey.__version__, str)
    assert spaceodyssey.__version__
