"""The twelve (scope, disposition) stop cases."""

import pytest

from tests.core.corpus import stop_grid
from tests.core.helpers import check_corpus_case

CASES = stop_grid
GRID = None


def _cases():
    global GRID
    if GRID is None:
        GRID = stop_grid()
    return GRID


def test_twelve_cases_with_unique_names():
    names = [c.name for c in _cases()]
    assert len(names) == 12 and len(set(names)) == 12


@pytest.mark.parametrize(
    "name",
    [
        f"stop_{s}_{d}"
        for s in ("lane", "asset", "program", "campaign")
        for d in ("close", "drain", "abort")
    ],
)
def test_stop_case(name):
    case = {c.name: c for c in _cases()}[name]
    check_corpus_case(case)
