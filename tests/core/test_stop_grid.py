"""The twelve (scope, disposition) stop cases.

Each case is run and checked by `test_corpus.py::test_corpus_three_way_parity`, which
covers the whole corpus, so only the grid's shape is pinned here.
"""

from tests.core.corpus import stop_grid


def test_twelve_cases_with_unique_names():
    names = [c.name for c in stop_grid()]
    assert len(names) == 12 and len(set(names)) == 12
