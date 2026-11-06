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


def test_scripted_acquisition_ids_do_not_collide_with_refresh_products():
    from spaceodyssey.studies.scripted import scripted_refresh_study

    study = scripted_refresh_study(ticks=(1,), values=(2.0,), ids=(4,), every=100)
    truths, roots = worlds(1)
    run = HostExecutor().run(
        study, base.index_world(truths, 0), roots[0], 30, refit=masked()
    )
    log = run.carry.log
    valid = np.flatnonzero(np.asarray(log.valid))
    ids = [int(log.acquisition_id[i]) for i in valid]
    assert len(ids) == len(set(ids)) and len(ids) >= 2, ids
    kinds = {int(log.kind[i]) for i in valid}
    assert toy.PRODUCT_KIND in kinds  # control: a product was written
    scripted = [i for i in valid if int(log.kind[i]) == toy.RECORD_MEASUREMENT]
    assert [int(log.acquisition_index[i]) for i in scripted] == [4]


def test_a_scripted_correction_still_finds_its_original():
    from spaceodyssey.core.enums import Audience
    from spaceodyssey.core.records import revision_set
    from spaceodyssey.studies.scripted import scripted_refresh_study

    study = scripted_refresh_study(
        ticks=(1, 2), values=(2.0, 9.0), ids=(4, 4), revisions=(0, 1), every=100
    )
    truths, roots = worlds(1)
    run = HostExecutor().run(
        study, base.index_world(truths, 0), roots[0], 30, refit=masked()
    )
    log = run.carry.log
    scripted = [
        i
        for i in np.flatnonzero(np.asarray(log.valid))
        if int(log.kind[i]) == toy.RECORD_MEASUREMENT
    ]
    assert len(scripted) == 2
    assert int(log.acquisition_id[scripted[0]]) == int(log.acquisition_id[scripted[1]])
    counts, replaced = revision_set(log, study.caps, Audience.OBSERVER, 0)
    assert [bool(counts[i]) for i in scripted] == [False, True]
    assert int(replaced[scripted[0]]) == 1


def _derived_measurement_study():
    import dataclasses as dc

    from spaceodyssey.core.capacities import Capacities
    from spaceodyssey.core.records import KindRegistry, RecordKind

    derived = dc.replace(toy.MEASUREMENT_KIND, derived=True)
    registry = KindRegistry((RecordKind("none", ()), derived, toy.AUXILIARY_KIND))
    return toy.survey_1(registry.lower(Capacities()))


def test_replay_skips_derived_rows_like_the_live_run():
    import jax.numpy as jnp

    from spaceodyssey.core import replay_belief
    from spaceodyssey.core.conformance import replay_beliefs

    study = _derived_measurement_study()
    truths, roots = worlds(2)
    run = HostExecutor().run_worlds(study, truths, roots, 40)
    live = np.asarray(run.carry.belief.n)
    assert live.sum() == 0  # control: nothing derived is folded live
    for w in range(2):
        carry = base.index_world(run.carry, w)
        trace = base.index_world(run.trace, w)
        assert int(jnp.sum(replay_belief(study, carry, trace).n)) == 0
    kit, _ = replay_beliefs(study, run, roots)
    assert np.asarray(kit.n).sum() == 0
