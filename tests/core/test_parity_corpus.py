"""A1: the compiled and host executors agree on every corpus case."""

import pytest

from spaceodyssey.core import (
    GO_TOLERANCE,
    KILL_TOLERANCE,
    CompiledExecutor,
    HostExecutor,
    histories_equal,
)
from tests.core.corpus import CORPUS
from tests.core.helpers import N_STEPS_CORPUS, service_of, worlds

N_WORLDS = 2


@pytest.mark.parametrize("case", CORPUS, ids=lambda c: c.name)
def test_ints_exact_and_floats_in_the_go_band(case):
    study = case.make_study()
    truths, roots = worlds(N_WORLDS)
    compiled = CompiledExecutor().run_worlds(
        study, truths, roots, N_STEPS_CORPUS, **service_of(case)
    )
    assert compiled.status.shape == (N_WORLDS,)
    for form in case.policy_forms:
        host = HostExecutor().run_worlds(
            study, truths, roots, N_STEPS_CORPUS, policy_form=form, **service_of(case)
        )
        assert host.status.tolist() == compiled.status.tolist(), (case.name, form)
        for w in range(N_WORLDS):
            rep = histories_equal(compiled.history(w), host.history(w))
            where = (case.name, form, w, rep)
            assert rep.structure_equal and rep.max_int_diff == 0, where
            assert rep.max_abs <= GO_TOLERANCE and rep.go, where
            assert not rep.vacuous, where


def test_tolerance_bands_are_ordered():
    assert GO_TOLERANCE < KILL_TOLERANCE == 1e-9
