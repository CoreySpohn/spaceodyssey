"""Derived kinds stay out of corrections; scripted pairs are unique (slow tier)."""

import dataclasses as dc

import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey import core
from spaceodyssey.core import CompiledExecutor, HostExecutor, handlers
from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.enums import I64, Audience, Capacity, EventKind, RunStatus
from spaceodyssey.core.events import empty_event_row
from spaceodyssey.core.records import KindRegistry, PayloadColumn, RecordKind
from spaceodyssey.studies import toy
from spaceodyssey.studies.scripted import ScriptedRecords
from tests.core.helpers import one_world, worlds

PUBLIC = Audience.OBSERVER | Audience.POLICY
SUMMARY = 3  # index of the derived kind in the registry below


def _registry(measurement_derived=False):
    measurement = dc.replace(toy.MEASUREMENT_KIND, derived=measurement_derived)
    summary = RecordKind("summary", (PayloadColumn("value", PUBLIC),), derived=True)
    return KindRegistry(
        (RecordKind("none", ()), measurement, toy.AUXILIARY_KIND, summary)
    )


def _study(kinds, revisions, ids=(4, 4), *, measurement_derived=False):
    process = ScriptedRecords(
        ticks=(1, 2),
        values=(2.0, 9.0),
        ids=ids,
        revisions=revisions,
        statuses=(0, 0),
        kinds=kinds,
    )
    caps = _registry(measurement_derived).lower(Capacities())
    return toy.survey_1(caps, policies=(toy.NullPolicy(),), process=process)


def _runs(study):
    truths, roots = worlds(2)
    host = HostExecutor().run_worlds(study, truths, roots, 20)
    compiled = CompiledExecutor().run(study, truths, roots, 20)
    return {"host": host, "compiled": compiled}


def _refused(run, w):
    log = run.carry.log
    mask = int(run.carry.overflow_mask[w])
    return int(np.asarray(log.valid[w]).sum()), mask


@pytest.mark.parametrize("executor", ["host", "compiled"])
def test_a_derived_correction_is_refused_not_logged(executor):
    # derived rev 0, then derived rev 1 of the same acquisition
    study = _study((SUMMARY, SUMMARY), (0, 1))
    run = _runs(study)[executor]
    for w in range(2):
        n_rows, mask = _refused(run, w)
        assert n_rows == 1
        assert mask & int(Capacity.INVALID_CORRECTION)
        assert not mask & int(Capacity.LOG_ROWS)
        assert int(run.status[w]) == RunStatus.OVERFLOW
        # the refused row is not counted in any release version
        assert int(run.carry.release_version[w, 0]) == 1


@pytest.mark.parametrize("executor", ["host", "compiled"])
@pytest.mark.parametrize("order", [(1, SUMMARY), (SUMMARY, 1)])
def test_a_correction_across_derived_and_source_is_refused(executor, order):
    study = _study(order, (0, 1))
    run = _runs(study)[executor]
    for w in range(2):
        n_rows, mask = _refused(run, w)
        assert n_rows == 1  # the first row stays, untouched
        assert mask & int(Capacity.INVALID_CORRECTION)
        log = run.carry.log
        assert int(log.superseded_by[w, 0]) == -1


@pytest.mark.parametrize("executor", ["host", "compiled"])
def test_source_corrections_and_distinct_derived_ids_are_not_refused(executor):
    # control: the same shapes with no derived involvement, and derived rows that
    # reuse nothing
    ok = _runs(_study((1, 1), (0, 1)))[executor]
    again = _runs(_study((SUMMARY, SUMMARY), (0, 0), ids=(4, 5)))[executor]
    for run, rows in ((ok, 2), (again, 2)):
        for w in range(2):
            n_rows, mask = _refused(run, w)
            assert n_rows == rows
            assert not mask & int(Capacity.INVALID_CORRECTION)


def test_scripted_pairs_must_be_unique():
    kw = dict(ticks=(1, 2), values=(2.0, 9.0), statuses=(0, 0))
    with pytest.raises(ValueError, match=r"\(4, 0\)"):
        ScriptedRecords(ids=(4, 4), revisions=(0, 0), **kw)
    # distinct revisions of one id, and one revision of distinct ids, are allowed
    ScriptedRecords(ids=(4, 4), revisions=(0, 1), **kw)
    ScriptedRecords(ids=(4, 5), revisions=(0, 0), **kw)


# ---- the release handler's supersession metadata ---------------------------------


def _release_pair(first, second, kinds, revisions=(0, 1)):
    """Release two rows of one acquisition id (revisions 0 and 1) in either order."""
    study = _study((1, 1), (0, 1))
    truth, root = one_world()
    carry = core.init_carry(study, truth)
    caps = study.caps
    n = caps.n_log
    pad = lambda v: jnp.asarray(list(v) + [0] * (n - 2), I64)  # noqa: E731
    log = dc.replace(
        carry.log,
        valid=jnp.arange(n) < 2,
        kind=pad(kinds),
        acquisition_id=pad([7, 7]),
        revision=pad(revisions),
        audience=pad([int(PUBLIC)] * 2),
        accepted_tick=pad([-1, -1]),
        released_tick=pad([-1, -1]),
        superseded_by=pad([-1, -1]),
    )
    carry = dc.replace(carry, log=log, n_log=jnp.asarray(2, I64))
    for slot in (first, second):
        event = dc.replace(
            empty_event_row(),
            kind=jnp.asarray(int(EventKind.RELEASE), I64),
            valid=jnp.asarray(True),
            tick=jnp.asarray(10, I64),
            slot=jnp.asarray(slot, I64),
            audience=jnp.asarray(int(PUBLIC), I64),
        )
        carry = handlers.h_release(
            dc.replace(carry, tick=event.tick), event, study, truth, root
        )[0]
    return carry.log


@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_release_never_supersedes_across_a_derived_row(order):
    for kinds in ([1, SUMMARY], [SUMMARY, 1]):
        log = _release_pair(*order, kinds)
        assert np.asarray(log.superseded_by)[:2].tolist() == [-1, -1], kinds
    # control: two revisions of one source row do supersede
    log = _release_pair(*order, [1, 1])
    assert np.asarray(log.superseded_by)[:2].tolist() == [1, -1]
    # control: two rows at one revision are not a supersession
    log = _release_pair(*order, [1, 1], revisions=(0, 0))
    assert np.asarray(log.superseded_by)[:2].tolist() == [-1, -1]
