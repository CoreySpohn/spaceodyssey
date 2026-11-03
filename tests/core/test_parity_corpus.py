"""A1: the compiled and host executors agree on every corpus case."""

import pytest

from spaceodyssey.core import (
    GO_TOLERANCE,
    KILL_TOLERANCE,
)
from tests.core.corpus import CORPUS
from tests.core.parity_helpers import corpus_parity_reports


@pytest.mark.parametrize("case", CORPUS, ids=lambda c: c.name)
def test_ints_exact_and_floats_in_the_go_band(case):
    for form, world, report in corpus_parity_reports(case):
        where = (case.name, form, world, report)
        assert report.structure_equal and report.max_int_diff == 0, where
        assert report.max_abs <= GO_TOLERANCE and report.go, where
        assert not report.vacuous, where


def test_tolerance_bands_are_ordered():
    assert GO_TOLERANCE < KILL_TOLERANCE == 1e-9
