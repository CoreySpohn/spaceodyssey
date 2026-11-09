"""Every invariant over every corpus case on both executors, with planted failures."""

import dataclasses
import types

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey.core import (
    Audience,
    EventKind,
    HostExecutor,
    histories_equal,
    index_world,
)
from spaceodyssey.core.conformance import (
    EXECUTORS,
    INVARIANTS,
    check_run,
    failures_of,
    make_worlds,
    probed,
    public_events,
    run_case,
)
from spaceodyssey.core.conformance.invariants import no_scorer_column_in_snapshot
from tests.core import record_helpers as rh
from tests.core.conformance_helpers import runs
from tests.core.corpus import CORPUS, CORPUS_BY_NAME
from tests.core.leak_helpers import leaky_wake

pytestmark = pytest.mark.clear_caches_between_tests


# Independent list: removing an invariant from check_run must break this file.
EXPECTED_INVARIANTS = (
    "order_keys_monotone",
    "exactly_once_settlement",
    "one_fold_per_acquisition",
    "no_violation",
    "balance_conserved",
    "unique_acquisition_id",
    "dense_acquisition_index",
    "release_not_before_available",
    "no_scorer_column_in_snapshot",
    "no_extra_wakes_under_private_events",
    "refit_inputs_equal_revision_set",
    "installs_match_product_rows",
)

# Independent of the runner: the corpus runs two worlds, numbered from zero.
EXPECTED_WORLDS = (0, 1)

# Independent of the study: the payload columns only the scorer may read, by record
# kind index (1 is the toy measurement, whose second column is the raw noise variate;
# 3 is the catalog record of its case). Every other column is readable by everyone.
SCORER_ONLY_COLUMNS = {"default": {1: (1,)}, "catalog_record": {1: (1,), 3: (1,)}}
N_KINDS, N_COLUMNS = 4, 4


def reference_layout(name):
    """The layout that judges snapshots, written here and not read from the study."""
    table = np.full((N_KINDS, N_COLUMNS), 7, dtype=np.int64)
    for kind, columns in SCORER_ONLY_COLUMNS.get(
        name, SCORER_ONLY_COLUMNS["default"]
    ).items():
        table[kind, list(columns)] = int(Audience.SCORER)
    return types.SimpleNamespace(payload_audience=table)


def corpus_failures(corpus_runs, name):
    """(world, failed results) of corpus runs judged against the independent layout."""
    out = []
    for run in corpus_runs:
        run = dataclasses.replace(run, reference_caps=reference_layout(name))
        results = check_run(run)
        assert tuple(result.name for result in results) == EXPECTED_INVARIANTS
        out.append((run.world, failures_of(results)))
    return out


@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("case", CORPUS, ids=lambda c: c.name)
def test_corpus_invariants(case, executor):
    """Fails if an invariant leaves check_run, a corpus case runs other worlds than 0
    and 1, or a run fails an invariant judged against the independent snapshot layout.
    """
    corpus_runs = runs(case.name, executor)
    worlds = sorted(run.world for run in corpus_runs)
    assert worlds == list(EXPECTED_WORLDS), (
        f"world IDs {worlds} differ from the expected {list(EXPECTED_WORLDS)}",
        case.name,
        executor,
    )
    for world, failures in corpus_failures(corpus_runs, case.name):
        assert not failures, (case.name, executor, world, failures)


@pytest.mark.parametrize("executor", EXECUTORS)
def test_the_snapshot_invariant_names_a_hidden_column_a_layout_shows(executor):
    """Fails if a layout that shows a hidden column passes the snapshot invariant, or
    the clean baseline run fails it.
    """
    run = runs("baseline", executor)[0]
    assert no_scorer_column_in_snapshot(run).passed
    broken = dataclasses.replace(
        run, study=rh.with_leaking_layout(run.study), reference_caps=run.study.caps
    )
    result = no_scorer_column_in_snapshot(broken)
    assert result.name == "no_scorer_column_in_snapshot"
    assert not result.passed and "snapshot shows a hidden column" in result.diagnostic


@pytest.mark.parametrize("numbers", [[], [0], [0, 0], [0, 2]])
def test_the_corpus_check_rejects_missing_duplicate_or_wrong_worlds(
    monkeypatch, numbers
):
    """Fails if a corpus run set with missing, repeated or renumbered worlds passes."""
    returned = [types.SimpleNamespace(world=world) for world in numbers]
    monkeypatch.setitem(globals(), "runs", lambda *args: returned)
    checks = [types.SimpleNamespace(name=n, passed=True) for n in EXPECTED_INVARIANTS]
    monkeypatch.setitem(globals(), "check_run", lambda run: checks)
    with pytest.raises(AssertionError, match="world"):
        test_corpus_invariants(CORPUS_BY_NAME["baseline"], "host")


@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("case", CORPUS, ids=lambda c: c.name)
def test_hand_history_and_case_checks_hold_on_both_executors(case, executor):
    for run in runs(case.name, executor):
        h = run.history
        assert h.n_records >= 1 and h.n_handled >= 2, (case.name, "vacuous")
        if case.hand_history is not None:
            assert h.events == case.hand_history, (case.name, executor)
        if case.hand_rows is not None:
            assert tuple(r[3:] for r in h.rows) == case.hand_rows, (case.name, executor)
        for k, check in enumerate(case.invariants):
            assert check(h), (case.name, executor, "case check", k)


@pytest.mark.parametrize("case", CORPUS, ids=lambda c: c.name)
def test_executors_agree_on_every_case(case):
    for host, compiled in zip(
        runs(case.name, "host"), runs(case.name, "compiled"), strict=True
    ):
        report = histories_equal(host.history, compiled.history)
        assert report.ok, (case.name, host.world, report)
        assert host.probe == compiled.probe, case.name


def test_every_new_case_has_a_hand_history_and_rows():
    new = {
        "coalesced_release",
        "bundle_unaffordable",
        "forbidden_effect",
        "catalog_record",
        "host_service",
        "revision",
    }
    assert new <= set(CORPUS_BY_NAME)
    for name in new:
        case = CORPUS_BY_NAME[name]
        assert case.hand_history is not None and case.hand_rows is not None
        assert len(case.hand_history) == len(case.hand_rows)
        assert case.invariants


def test_the_probe_adds_private_events_that_were_handled():
    # the probe is not vacuous: its events are handled, and they are not in the rows
    # the comparison keeps
    case = CORPUS_BY_NAME["baseline"]
    (run,) = run_case(case, "host", n_worlds=1)
    study = probed(case.make_study())
    truths, roots = make_worlds(study, 1)
    _, trace = HostExecutor().run(study, index_world(truths, 0), roots[0], 40)
    handled = np.asarray(trace.handled)
    private = np.asarray(trace.kind)[handled] == EventKind.PROCESS
    assert int(private.sum()) == 10
    assert run.probe[2] == public_events(run.trace) == public_events(trace)


# ---- planted violations: each invariant fails on a run that breaks it ---------------


def planted(run, selector, value):
    """The run with one leaf of its carry replaced."""
    return dataclasses.replace(run, carry=eqx.tree_at(selector, run.carry, value))


def planted_trace(run, selector, value):
    """The run with one leaf of its trace replaced."""
    return dataclasses.replace(run, trace=eqx.tree_at(selector, run.trace, value))


def _log_set(run, name, i, value):
    column = getattr(run.carry.log, name)
    return planted(run, lambda c: getattr(c.log, name), column.at[i].set(value))


def _swap_order(run):
    tick = run.trace.tick
    handled = np.flatnonzero(np.asarray(run.trace.handled))
    return planted_trace(run, lambda t: t.tick, tick.at[handled[1]].set(10**6))


def _double_settle(run):
    return planted(run, lambda c: c.double_settle, jnp.asarray(True))


def _dup_fold(run):
    return planted(run, lambda c: c.dup_fold, jnp.asarray(True))


def _extra_fold(run):
    return planted(run, lambda c: c.belief.n, run.carry.belief.n + 5)


def _violation(run):
    return planted(run, lambda c: c.violation, jnp.asarray(True))


def _lost_funds(run):
    return planted(run, lambda c: c.ledger.balance, run.carry.ledger.balance + 0.5)


def _same_id(run):
    ids = np.asarray(run.carry.log.acquisition_id)
    return _log_set(run, "acquisition_id", 1, int(ids[0]))


def _index_gap(run):
    return _log_set(run, "acquisition_index", 0, 7)


def _repeated_index(run):
    index = np.asarray(run.carry.log.acquisition_index)
    subject = np.asarray(run.carry.log.subject)
    # a second row of the first subject takes the first row's index
    other = int(np.flatnonzero(subject[1:] == subject[0])[0]) + 1
    return _log_set(run, "acquisition_index", other, int(index[0]))


def _early_release(run):
    log = run.carry.log
    return _log_set(run, "released_tick", 0, int(log.available_tick[0]) - 1)


def _leaking_layout(run):
    study = rh.with_leaking_layout(run.study)
    return dataclasses.replace(run, study=study, reference_caps=run.study.caps)


def _first_product_slot(run):
    from spaceodyssey.core.conformance.invariants import _product_slots

    return int(_product_slots(run)[0])


def _wrong_pedigree(run):
    slot = _first_product_slot(run)
    product = run.carry.log.product
    return planted(
        run, lambda c: c.log.product, product.at[slot, 4].add(1)
    )  # a triple's acquisition index


def _dropped_pedigree_row(run):
    slot = _first_product_slot(run)
    product = run.carry.log.product
    return planted(run, lambda c: c.log.product, product.at[slot, 2].add(-1))


def _log_length_index(run):
    """A product indexed by its log position, not by its checkpoint's grid ordinal."""
    slot = _first_product_slot(run)
    return _log_set(run, "acquisition_index", slot, slot + 40)


def _wrong_recorded_status(run):
    return planted(run, lambda c: c.refresh_status, run.carry.refresh_status + 3)


def _wrong_product_channel(run):
    return _log_set(run, "channel", _first_product_slot(run), 5)


PLANTED = {
    "order_keys_monotone": (_swap_order,),
    "exactly_once_settlement": (_double_settle, _same_id),
    "one_fold_per_acquisition": (_dup_fold, _extra_fold),
    "no_violation": (_violation,),
    "balance_conserved": (_lost_funds,),
    "unique_acquisition_id": (_same_id,),
    "dense_acquisition_index": (_index_gap, _repeated_index),
    "release_not_before_available": (_early_release,),
    "no_scorer_column_in_snapshot": (_leaking_layout,),
}
# invariants that only mean something on a refresh-bearing case
ON_REFRESH = {
    "refit_inputs_equal_revision_set": (_wrong_pedigree, _dropped_pedigree_row),
    "installs_match_product_rows": (
        _wrong_recorded_status,
        _wrong_product_channel,
        _log_length_index,
    ),
}


@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("name", sorted(PLANTED))
def test_each_invariant_catches_its_planted_violation(name, executor):
    inv = next(f for f in INVARIANTS if f.__name__ == name)
    run = runs("baseline", executor)[0]
    assert inv(run), "the unplanted run must pass"
    for plant in PLANTED[name]:
        result = inv(plant(run))
        assert not result.passed and result.diagnostic, (name, plant.__name__)


@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("name", sorted(ON_REFRESH))
def test_refresh_invariants_catch_their_planted_violation(name, executor):
    inv = next(f for f in INVARIANTS if f.__name__ == name)
    run = runs("refresh", executor)[0]
    assert inv(run), "the unplanted run must pass"
    for plant in ON_REFRESH[name]:
        result = inv(plant(run))
        assert not result.passed and result.diagnostic, (name, plant.__name__)


def test_every_invariant_has_a_planted_violation_or_a_mutant():
    # the wake invariant's negative control is a wake rule that ignores audience
    assert {f.__name__ for f in INVARIANTS} == set(PLANTED) | set(ON_REFRESH) | {
        "no_extra_wakes_under_private_events"
    }


@pytest.mark.parametrize("executor", EXECUTORS)
def test_a_wake_rule_that_ignores_audience_adds_wakes(executor):
    case = CORPUS_BY_NAME["baseline"]
    (run, *_) = run_case(case, executor, n_worlds=1, probe_wake_fn=leaky_wake)
    from spaceodyssey.core.conformance.invariants import (
        no_extra_wakes_under_private_events,
    )

    result = no_extra_wakes_under_private_events(run)
    assert not result.passed and result.diagnostic


def test_an_unprobed_run_cannot_pass_the_wake_invariant():
    from spaceodyssey.core.conformance.invariants import (
        no_extra_wakes_under_private_events,
    )

    (run,) = run_case(CORPUS_BY_NAME["baseline"], "compiled", n_worlds=1, probe=False)
    assert not no_extra_wakes_under_private_events(run).passed


def test_overflow_cases_are_judged_on_what_they_kept():
    run = runs("overflow_log", "compiled")[0]
    from spaceodyssey.core.conformance.invariants import balance_conserved

    result = balance_conserved(run)
    assert result.passed and result.detail["overflow"] is True
    assert "skipped" not in result.detail
