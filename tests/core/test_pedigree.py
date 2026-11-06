"""Pedigree and producer-local indices on runs (slow tier)."""

import numpy as np

from spaceodyssey.core import HostExecutor
from spaceodyssey.core.executors import base
from spaceodyssey.core.records import pedigree_of
from spaceodyssey.studies import toy
from tests.core.helpers import worlds
from tests.core.refresh_helpers import (
    masked,
    oracle_refresh_study,
    refresh_study,
    run_pair,
)

SERVICE_CHANNEL = 0  # QueryPolicy.service default
QUERY_DECISION = 1  # oracle_refresh_study query_at


def _product_slots(carry):
    log = carry.log
    keep = np.asarray(log.valid) & (np.asarray(log.kind) == toy.PRODUCT_KIND)
    return np.flatnonzero(keep).tolist()


def _run_refresh(n_steps=60):
    h, c = run_pair(refresh_study(100), n_steps, 2, 7)
    return h, c


def test_product_rows_name_the_rows_they_were_computed_from():
    h, _ = _run_refresh()
    carry = base.index_world(h.carry, 0)
    slots = _product_slots(carry)
    assert slots
    log = carry.log
    measured = {
        (int(log.channel[i]), int(log.acquisition_index[i]), int(log.revision[i]))
        for i in np.flatnonzero(np.asarray(log.valid))
        if int(log.kind[i]) == toy.RECORD_MEASUREMENT
    }
    for slot in slots:
        audience, _, triples = pedigree_of(log, slot)
        assert audience in (1, 2)
        assert set(triples) <= measured
        assert list(triples) == sorted(triples)


def test_refresh_products_are_indexed_by_grid_ordinal_with_revision_zero():
    h, _ = _run_refresh()
    carry = base.index_world(h.carry, 0)
    log = carry.log
    slots = _product_slots(carry)
    # checkpoints with nothing dirty write no row, so ordinals may skip; the index is
    # the position on the grid (every=100 here), never a count of log rows
    for s in slots:
        assert int(log.acquisition_index[s]) * 100 == int(log.available_tick[s])
    assert {int(log.revision[s]) for s in slots} == {0}
    assert {int(log.channel[s]) for s in slots} <= set(toy.PRODUCT_CHANNELS)


def test_service_answers_carry_the_decision_as_index():
    study, service = oracle_refresh_study(130)
    truths, roots = worlds(1)
    run = HostExecutor().run(
        study,
        base.index_world(truths, 0),
        roots[0],
        40,
        service=service,
        refit=masked(),
    )
    log = run.carry.log
    answers = [
        i
        for i in np.flatnonzero(np.asarray(log.valid))
        if int(log.kind[i]) not in (toy.RECORD_MEASUREMENT, toy.PRODUCT_KIND)
        and int(log.channel[i]) == SERVICE_CHANNEL
    ]
    assert answers
    # the Query's decision is the answer's acquisition index, not a global counter
    assert [int(log.acquisition_index[i]) for i in answers] == [QUERY_DECISION] * len(
        answers
    )


def test_a_log_length_index_for_answers_is_caught(monkeypatch):
    """Control: an answer indexed by the log length instead of the decision differs."""
    import dataclasses

    import equinox as eqx
    import jax

    from spaceodyssey.core import service as svc

    real = svc.write_log

    def by_length(carry, rec, caps):
        rec = dataclasses.replace(rec, acquisition_index=carry.n_log + 7)
        return real(carry, rec, caps)

    monkeypatch.setattr(svc, "write_log", by_length)
    eqx.clear_caches()
    jax.clear_caches()
    try:
        study, service = oracle_refresh_study(130)
        truths, roots = worlds(1)
        run = HostExecutor().run(
            study,
            base.index_world(truths, 0),
            roots[0],
            40,
            service=service,
            refit=masked(),
        )
        log = run.carry.log
        idx = [
            int(log.acquisition_index[i])
            for i in np.flatnonzero(np.asarray(log.valid))
            if int(log.channel[i]) == SERVICE_CHANNEL
            and int(log.kind[i]) not in (toy.RECORD_MEASUREMENT, toy.PRODUCT_KIND)
        ]
        assert idx and idx != [QUERY_DECISION] * len(idx)
    finally:
        eqx.clear_caches()
        jax.clear_caches()
