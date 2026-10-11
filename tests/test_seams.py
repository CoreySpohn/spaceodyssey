"""Tests for the fixed seam vocabulary."""

from spaceodyssey.seams import DATA_PLANE_EDGES, KNOWN_OPERATORS, SEAMS


def test_ten_unique_seams():
    """The seam vocabulary has exactly ten unique names."""
    assert len(SEAMS) == 10
    assert len(set(SEAMS)) == 10


def test_edges_reference_known_seams():
    """Every data-plane edge endpoint is a declared seam."""
    for producer, consumer in DATA_PLANE_EDGES:
        assert producer in SEAMS
        assert consumer in SEAMS
        assert producer != consumer


def test_known_operators():
    """The three known outcome operators are declared."""
    assert KNOWN_OPERATORS == {"sample", "propagate", "integrate"}
