"""Corpus parity runs shared by the acceptance tests and close-out gate."""

from spaceodyssey.core import CompiledExecutor, HostExecutor, histories_equal
from tests.core.helpers import N_STEPS_CORPUS, service_of, worlds

N_WORLDS = 2


def corpus_parity_reports(case):
    """Compare all worlds and host policy forms, including run statuses."""
    reports = []
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
            reports.append((form, w, rep))

    return reports
