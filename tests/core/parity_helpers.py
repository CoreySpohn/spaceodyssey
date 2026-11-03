"""Corpus parity runs shared by the acceptance tests and close-out gate."""

import functools

from spaceodyssey.core import CompiledExecutor, HostExecutor, histories_equal
from spaceodyssey.studies.conformance import CORPUS_BY_NAME
from tests.core.helpers import N_STEPS_CORPUS, service_of, worlds

N_WORLDS = 2
N_RUN_WORLDS = 4


@functools.cache
def corpus_runs(name):
    """One compiled run and one host run per policy form of a corpus case.

    Both cover `N_RUN_WORLDS` worlds; the parity reports use the first `N_WORLDS`.
    A run is a pure function of the case, so every consumer shares these.
    """
    case = CORPUS_BY_NAME[name]
    study = case.make_study()
    truths, roots = worlds(N_RUN_WORLDS)
    compiled = CompiledExecutor().run_worlds(
        study, truths, roots, N_STEPS_CORPUS, **service_of(case)
    )
    hosts = {
        form: HostExecutor().run_worlds(
            study, truths, roots, N_STEPS_CORPUS, policy_form=form, **service_of(case)
        )
        for form in case.policy_forms
    }
    return compiled, hosts


@functools.cache
def corpus_parity_reports(case):
    """Compare the first worlds and every host policy form, including run statuses."""
    compiled, hosts = corpus_runs(case.name)
    assert compiled.status.shape == (N_RUN_WORLDS,)
    reports = []
    for form, host in hosts.items():
        assert host.status.tolist() == compiled.status.tolist(), (case.name, form)
        for w in range(N_WORLDS):
            rep = histories_equal(compiled.history(w), host.history(w))
            reports.append((form, w, rep))
    return reports


def check_corpus_case(case):
    """Three-way check of one corpus case over the shared runs.

    Scan, host (each policy form), hand history, invariants and non-vacuity, on every
    world of the run.
    """
    compiled, hosts = corpus_runs(case.name)
    for w in range(N_RUN_WORLDS):
        h = compiled.history(w)
        assert h.n_records >= 1 and h.n_handled >= 2, (case.name, "vacuous")
        for form in case.policy_forms:
            r = histories_equal(h, hosts[form].history(w))
            assert r.structure_equal and r.max_int_diff == 0, (case.name, form, r)
            assert r.max_abs <= 1e-9, (case.name, form, r)
        if case.hand_history is not None:
            assert h.events == case.hand_history, case.name
        if case.hand_rows is not None:
            assert tuple(r[3:] for r in h.rows) == case.hand_rows, case.name
        for k, inv in enumerate(case.invariants):
            assert inv(h), (case.name, "invariant", k)
