"""Admission rules: permitted effects, atomic groups, unsupported kinds."""

import dataclasses

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey import core
from spaceodyssey.core.enums import (
    ALL_EFFECTS,
    AdmissionCode,
    EffectKind,
    effect_mask,
)
from spaceodyssey.core.executors.base import (
    canonical_history,
    histories_equal,
    index_world,
)
from spaceodyssey.core.executors.compiled import CompiledExecutor
from spaceodyssey.core.executors.host import HostExecutor
from spaceodyssey.studies import builder
from spaceodyssey.studies import toy as kinds
from tests.core.corpus import CORPUS_BY_NAME
from tests.core.helpers import CAPS, admit_j, effects_of, one_world, worlds

TRUTH, ROOT = one_world()


def carry0(study):
    return eqx.filter_jit(core.init_carry)(study, TRUTH)


def observe_rows(subjects, channel_indices, groups):
    e = core.effects_none(CAPS)
    return dataclasses.replace(
        e,
        kind=jnp.full(CAPS.n_effects, EffectKind.OBSERVE, jnp.int64),
        subject=jnp.asarray(subjects, jnp.int64),
        channel_index=jnp.asarray(channel_indices, jnp.int64),
        group=jnp.asarray(groups, jnp.int64),
    )


def with_program_funds(study, funds):
    """Set the program row's balance of resource 1."""
    row = study.caps.n_assets
    return eqx.tree_at(lambda s: s.budget, study, study.budget.at[row, 1].set(funds))


# ---- permitted effects ---------------------------------------------------------


def test_forbidden_effect_refused():
    only_observe = builder.survey_1(
        permitted_effects=(effect_mask(EffectKind.OBSERVE),)
    )
    c0 = carry0(only_observe)
    stop = effects_of(EffectKind.STOP)
    c1, codes = admit_j(c0, stop, only_observe)
    assert int(codes[0]) == AdmissionCode.FORBIDDEN
    r = core.tree_equal_report(c0, c1)
    assert r["max_int_diff"] == 0 and r["max_abs"] == 0.0
    # control: the same effect under a mask that permits it is admitted and queued
    open_study = builder.survey_1(permitted_effects=(ALL_EFFECTS,))
    c2, codes = admit_j(carry0(open_study), stop, open_study)
    assert int(codes[0]) == AdmissionCode.OK
    assert int(c2.ev.valid.sum()) == int(c0.ev.valid.sum()) + 1


def test_permitted_effect_is_admitted_under_a_narrow_mask():
    only_observe = builder.survey_1(
        permitted_effects=(effect_mask(EffectKind.OBSERVE),)
    )
    _c1, codes = admit_j(
        carry0(only_observe), observe_rows([0, 0], [0, 0], [0, 1]), only_observe
    )
    assert int(codes[0]) == AdmissionCode.OK


def test_each_phase_has_its_own_mask():
    wait_then_observe = builder.survey_1(
        policies=(kinds.NullPolicy(), kinds.NullPolicy()),
        phase_until=(300, 1000),
        permitted_effects=(effect_mask(EffectKind.WAIT), effect_mask(EffectKind.STOP)),
    )
    c0 = carry0(wait_then_observe)
    _, codes = admit_j(c0, effects_of(EffectKind.STOP), wait_then_observe)
    assert int(codes[0]) == AdmissionCode.FORBIDDEN
    later = eqx.tree_at(lambda c: c.tick, c0, jnp.asarray(400, jnp.int64))
    _, codes = admit_j(later, effects_of(EffectKind.STOP), wait_then_observe)
    assert int(codes[0]) == AdmissionCode.OK


def test_one_mask_per_phase_is_required():
    with pytest.raises(ValueError):
        builder.survey_1(permitted_effects=(ALL_EFFECTS, ALL_EFFECTS))


# ---- Allocate requires awards -------------------------------------------------


def test_allocate_answers_unsupported_and_changes_nothing():
    """Fails if Allocate changes an awards-off carry or fails to credit a valid awards-
    on recipient.
    """
    study = builder.survey_1()
    c0 = carry0(study)
    c1, codes = admit_j(c0, effects_of(EffectKind.ALLOCATE), study)
    assert int(codes[0]) == AdmissionCode.UNSUPPORTED
    r = core.tree_equal_report(c0, c1)
    assert r["max_int_diff"] == 0 and r["max_abs"] == 0.0

    from tests.core.award_helpers import award_study

    enabled = award_study((), (), caps_extra={"snapshot_balances": True})
    before = carry0(enabled)
    effects = core.effects_none(enabled.caps)
    effects = dataclasses.replace(
        effects,
        kind=effects.kind.at[0].set(EffectKind.ALLOCATE),
        amount=effects.amount.at[0, 1].set(5.0),
        recipient=effects.recipient.at[0].set(2),
        window_from=effects.window_from.at[0].set(30),
        window_until=effects.window_until.at[0].set(200),
    )
    after, accepted = admit_j(before, effects, enabled)
    assert int(accepted[0]) == AdmissionCode.OK
    expected = before.ledger.balance.at[2, 1].add(5.0)
    assert jnp.array_equal(after.ledger.balance, expected)
    assert float(after.ledger.granted[2, 1]) == 5.0


# ---- atomic groups -------------------------------------------------------------


def test_bundle_atomic():
    # the program row holds one unit of the second resource; the bundle needs two
    study = with_program_funds(builder.survey_1(), 1.0)
    c0 = carry0(study)
    bundle = observe_rows([1, 1], [0, 2], [5, 5])
    c1, codes = admit_j(c0, bundle, study)
    assert [int(x) for x in codes] == [AdmissionCode.NO_FUNDS] * 2
    assert np.array_equal(c1.acquisition_counts, c0.acquisition_counts)
    assert np.array_equal(c1.ledger.balance, c0.ledger.balance)
    assert np.array_equal(c1.ledger.lane_busy_until, c0.ledger.lane_busy_until)
    assert np.array_equal(c1.ledger.lane_gen, c0.ledger.lane_gen)
    assert int(c1.next_sid) == int(c0.next_sid)
    assert int(c1.ev.valid.sum()) == int(c0.ev.valid.sum())
    # control: with enough funds both rows are admitted and take indices 0 and 1
    rich = with_program_funds(builder.survey_1(), 2.0)
    c2, codes = admit_j(carry0(rich), bundle, rich)
    assert [int(x) for x in codes] == [AdmissionCode.OK] * 2
    assert int(c2.acquisition_counts[1]) == 2
    done = np.asarray(c2.ev.valid & (c2.ev.kind == core.EventKind.DONE))
    assert sorted(np.asarray(c2.ev.acquisition_index)[done].tolist()) == [0, 1]


def test_separate_groups_are_admitted_one_by_one():
    study = with_program_funds(builder.survey_1(), 1.0)
    rows = observe_rows([1, 1], [0, 2], [5, 6])
    c1, codes = admit_j(carry0(study), rows, study)
    assert [int(x) for x in codes] == [AdmissionCode.OK, AdmissionCode.NO_FUNDS]
    assert int(c1.acquisition_counts[1]) == 1


def test_group_refusal_names_the_first_refused_row():
    # lane 0 is busy for the first row; the second row alone would pass
    study = builder.survey_1()
    busy = eqx.tree_at(
        lambda c: c.ledger.lane_busy_until,
        carry0(study),
        jnp.asarray([50, 0, 0], jnp.int64),
    )
    _, codes = admit_j(busy, observe_rows([1, 1], [0, 2], [5, 5]), study)
    assert [int(x) for x in codes] == [AdmissionCode.LANE_BUSY] * 2


def test_group_on_one_lane_is_refused_whole():
    study = builder.survey_1()
    c0 = carry0(study)
    c1, codes = admit_j(c0, observe_rows([0, 1], [0, 0], [5, 5]), study)
    assert [int(x) for x in codes] == [AdmissionCode.LANE_BUSY] * 2
    assert int(c1.acquisition_counts.sum()) == 0


def test_group_needs_queue_room_for_every_row():
    study = builder.survey_1()
    c0 = carry0(study)
    free = ~np.asarray(c0.ev.valid)
    keep = np.flatnonzero(free)[:1]  # leave exactly one free slot
    valid = np.ones_like(free)
    valid[keep] = False
    crowded = eqx.tree_at(lambda c: c.ev.valid, c0, jnp.asarray(valid))
    c1, codes = admit_j(crowded, observe_rows([1, 1], [0, 2], [5, 5]), study)
    assert [int(x) for x in codes] == [AdmissionCode.NO_SLOT] * 2
    assert int(c1.acquisition_counts.sum()) == 0


def _bundle_run(study):
    truths, roots = worlds(2)
    cs, ts = CompiledExecutor().run(study, truths, roots, 40)
    out = []
    for w in range(2):
        layout = study.observer.layout(study.caps)
        h = canonical_history(index_world(cs, w), index_world(ts, w), layout)
        host_carry, th = HostExecutor().run(study, index_world(truths, w), roots[w], 40)
        parity = histories_equal(h, canonical_history(host_carry, th, layout))
        assert parity.structure_equal and parity.max_int_diff == 0
        out.append((h, index_world(cs, w)))
    return out


def test_corpus_bundle_with_enough_funds_admits_both_rows():
    for h, _ in _bundle_run(CORPUS_BY_NAME["bundle"].make_study()):
        assert h.n_records == 6
        assert not np.any(h.ints["admit_code"] == AdmissionCode.NO_FUNDS)


def test_corpus_bundle_without_enough_funds_admits_neither_row():
    # one unit is spent by the first observation; the bundle then finds 1.5 of the 2
    study = with_program_funds(CORPUS_BY_NAME["bundle"].make_study(), 2.5)
    for h, carry in _bundle_run(study):
        bundle_codes = h.ints["admit_code"][2]
        assert [int(x) for x in bundle_codes] == [AdmissionCode.NO_FUNDS] * 2
        assert int(carry.acquisition_counts[1]) == 0
        assert h.n_records == 1
