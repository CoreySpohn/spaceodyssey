"""Parity requires matching types, exact integers, and matching NaN positions."""

import numpy as np
import pytest

from spaceodyssey.core.executors.base import ParityReport, tree_equal_report


def _parity(a, b):
    report = tree_equal_report(a, b)
    return ParityReport(
        structure_equal=report["structure_equal"] and report["shapes_equal"],
        max_int_diff=report["max_int_diff"],
        max_abs=report["max_abs"],
        max_rel=report["max_rel"],
        max_ulps=report["max_ulps"],
        vacuous=False,
    )


@pytest.mark.parametrize(
    "a,b",
    [
        (np.array([0], np.int64), np.array([-(2**63)], np.int64)),
        (np.array([0], np.uint64), np.array([2**64 - 1], np.uint64)),
        (np.array([1.0]), np.array([np.nan])),
        (np.array([np.nan]), np.array([1.0])),
        (np.array([1], np.int64), np.array([1.0], np.float64)),
        (np.array([1], np.int32), np.array([1], np.int64)),
        (np.array([1.0], np.float32), np.array([1.0], np.float64)),
        (np.array([1.0]), np.array([[1.0]])),
        (np.array([], np.int64), np.array([], np.float64)),
        (np.array([np.inf]), np.array([-np.inf])),
    ],
)
def test_unequal_values_cannot_pass_parity(a, b):
    report = _parity(a, b)
    assert not report.ok and not report.go


@pytest.mark.parametrize(
    "a",
    [
        np.array([np.nan, np.inf, -np.inf, 1.0]),
        np.array([-(2**63), 0, 2**63 - 1], np.int64),
        np.array([0, 2**64 - 1], np.uint64),
        np.array([True, False]),
        np.array([], np.float64),
    ],
)
def test_equal_values_pass_parity(a):
    assert _parity(a, a.copy()).go


@pytest.mark.parametrize(
    "a,b",
    [
        (np.array([1 + 2j]), np.array([1 + 3j])),
        (np.array([1 + 2j]), np.array([1.0 + 0j])),
        (np.array([1.0 + 0j], np.complex64), np.array([1.0 + 0j], np.complex64)),
        (np.array([1.0]), np.array([1 + 2j])),
    ],
)
def test_complex_leaves_are_rejected_not_compared_by_real_part(a, b):
    with pytest.raises(TypeError, match="complex"):
        tree_equal_report(a, b)
