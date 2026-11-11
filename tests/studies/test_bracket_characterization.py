"""Characterization in the bracket policy: retries, retirement, `n_char`, the reserve.

The decision tests hand the policy a summary and a state and read one decision, so no
study runs; the run test checks the block log against the outcome table.
"""

import dataclasses
import functools
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey.adapters.photomancy import N_GRID, DecisionSummary
from spaceodyssey.core.enums import EffectKind
from spaceodyssey.studies import bracket as br

from .bracket_helpers import N_SLOTS, build

CLASS_ONE_CODE = br.CLASS_ONE + br.REVEAL_OFFSET
POLICY_REASON = "compiles the bracket policy decision program"
N_STARS = 2


@functools.cache
def _built():
    return build(run=False)


def _state(n_char=1, retire_after=br.RETIRE_AFTER, **slot):
    """A policy state: kickoff done, and slot 0 of star 0 holding `slot` fields."""
    n = N_STARS
    pstate = np.zeros(br.pstate_size(n, N_SLOTS, n_char, retire_after))
    pstate[n * (3 + N_SLOTS) + 1] = n
    ext = n * (3 + N_SLOTS) + 3
    fields = ("att", "cons", "seen", "last_mode", "last_ord", "floor")
    for name, value in slot.items():
        if name == "n_blk":
            pstate[ext + n * N_SLOTS * br.SLOT_FIELDS] = value
        else:
            pstate[ext + fields.index(name)] = value
    return jnp.asarray(pstate)


def _decide(pstate=None, tick_d=0.0, policy_kw=None, survey=None, **leaves):
    """`(effects, new state)` of one decision on a candidate at slot 0 of star 0."""
    _, study, _, _, _, _ = _built()
    policy = dataclasses.replace(
        study.agenda.policies[0], budget_d=1000.0, **(policy_kw or {})
    )
    if survey is not None:
        study = dataclasses.replace(
            study,
            channel_model=dataclasses.replace(study.channel_model, survey=survey),
        )
    n = N_STARS
    belief = DecisionSummary.zeros(n, N_SLOTS, N_GRID, br.N_DETECT, br.N_BLOCK, n_an=1)
    base = {
        "n_det": belief.n_det.at[0, 0].set(3),
        "p_class": belief.p_class.at[0, 0].set(0.6),
        "status": jnp.zeros_like(belief.status),
        "p_block": belief.p_block.at[0, 0].set(1.0),
        "p_det": belief.p_det.at[0, 0].set(1.0),
    }
    base.update(leaves)
    belief = dataclasses.replace(belief, **base)
    if pstate is None:
        pstate = _state()
    view = types.SimpleNamespace(
        belief=belief,
        tick=jnp.asarray(round(tick_d * 86_400_000)),
        n_dec=jnp.asarray(1),
    )
    effects, new, _ = policy.decide(pstate, view, study, jax.random.key(0))
    return effects, np.asarray(new)


def _slot0(new):
    """The block state of slot 0 of star 0 after a decision."""
    ext = N_STARS * (3 + N_SLOTS) + 3
    return new[ext : ext + br.SLOT_FIELDS]


def _chose_block(effects):
    return int(effects.channel_index[0]) >= br.N_DETECT


def _at(value, dtype=jnp.int64):
    """A ``[n_stars, n_slots]`` array that is `value` at slot 0 of star 0."""
    return jnp.zeros((N_STARS, N_SLOTS), dtype).at[0, 0].set(value)


def _read(n):
    """The `n_blocks` leaf: `n` block rows read on star 0."""
    return jnp.zeros(N_STARS, jnp.int64).at[0].set(n)


@pytest.mark.medium(reason=POLICY_REASON)
def test_a_block_that_fell_short_is_retried_at_a_longer_mode():
    """Once its row is read with no spectrum, the slot is blocked again, longer.

    Fails if a failed block still locks the slot, if the retry reuses the failed mode
    (or a shorter one), or if it starts before a refit has read the failed block.
    """
    failed = _state(att=1, last_mode=1, last_ord=1, n_blk=1)
    effects, new = _decide(failed, n_blocks=_read(1))
    assert int(effects.channel_index[0]) == br.N_DETECT + 2
    att, cons, _, mode, ordinal, floor = _slot0(new)
    assert (att, cons, mode, ordinal, floor) == (2, 1, 2, 2, 2)
    unread, _ = _decide(failed, n_blocks=_read(0))
    assert not _chose_block(unread), "a block was started before its row was read"
    first, new = _decide(_state())
    assert int(first.channel_index[0]) == br.N_DETECT, "a first block takes mode 0"
    assert _slot0(new)[0] == 1


@pytest.mark.medium(reason=POLICY_REASON)
def test_three_failures_in_a_row_retire_the_slot():
    """The third consecutive failed block ends the attempts; two do not."""
    two = _state(att=2, cons=1, last_mode=1, last_ord=2, n_blk=2)
    effects, _ = _decide(two, n_blocks=_read(2))
    assert _chose_block(effects), "two failures retire the slot"
    three = _state(att=3, cons=2, last_mode=1, last_ord=3, n_blk=3)
    effects, new = _decide(three, n_blocks=_read(3))
    assert not _chose_block(effects), "the slot was not retired after three failures"
    assert _slot0(new)[0] == 3
    # a policy asked to retire after one failure reproduces the single block per slot
    once = _state(retire_after=1, att=1, last_ord=1, n_blk=1)
    effects, _ = _decide(once, policy_kw={"retire_after": 1}, n_blocks=_read(1))
    assert not _chose_block(effects)


@pytest.mark.medium(reason=POLICY_REASON)
def test_n_char_two_needs_two_delivered_spectra():
    """A class-1 slot with one spectrum is blocked again at n_char 2, not at 1.

    Fails if the slot closes at the first reveal under n_char 2, if it stays open
    after the second spectrum, or if a failure is read from a delivered spectrum.
    """
    kw = {"n_char": 2}
    one = _state(n_char=2, att=1, seen=0, last_ord=1, n_blk=1)
    seen_one = {
        "revealed": _at(CLASS_ONE_CODE),
        "n_spectra": _at(1),
        "n_blocks": _read(1),
    }
    effects, new = _decide(one, policy_kw=kw, **seen_one)
    assert _chose_block(effects) and int(effects.channel_index[0]) == br.N_DETECT
    # the first spectrum is a success: no failure is carried into the second block
    assert _slot0(new)[1] == 0 and _slot0(new)[5] == 0
    effects, _ = _decide(one, policy_kw={"n_char": 1}, **seen_one)
    assert not _chose_block(effects), "n_char 1 is done after one spectrum"
    two = _state(n_char=2, att=2, seen=1, last_ord=2, n_blk=2)
    done = {"revealed": _at(CLASS_ONE_CODE), "n_spectra": _at(2), "n_blocks": _read(2)}
    effects, _ = _decide(two, policy_kw=kw, **done)
    assert not _chose_block(effects)


def test_a_slot_is_confirmed_at_the_configured_n_char():
    """Host arrays: class 1 revealed with fewer spectra than n_char is not confirmed."""
    belief = types.SimpleNamespace(
        revealed=np.asarray([[CLASS_ONE_CODE, CLASS_ONE_CODE, br.REVEAL_OFFSET]]),
        n_spectra=np.asarray([[1, 2, 2]]),
    )
    assert np.array_equal(np.asarray(br.confirmed_slots(belief))[0], [1, 1, 0])
    assert np.array_equal(np.asarray(br.confirmed_slots(belief, 2))[0], [0, 1, 0])


def _search_case(reserve_d, budget_d):
    """A decision with one track of slot 0 (a detection) and a search open on star 1."""
    p_new = jnp.zeros((N_STARS, N_GRID, br.N_DETECT)).at[1, :, 0].set(0.5)
    leaves = {
        "n_det": _at(1),
        "p_det": jnp.zeros((N_STARS, N_SLOTS, N_GRID, br.N_DETECT)),
        "p_new": p_new,
    }
    return _decide_budget({"reserve_d": reserve_d, "budget_d": budget_d}, **leaves)


def _decide_budget(kw, **leaves):
    _, study, _, _, _, _ = _built()
    policy = dataclasses.replace(study.agenda.policies[0], **kw)
    n = N_STARS
    belief = DecisionSummary.zeros(n, N_SLOTS, N_GRID, br.N_DETECT, br.N_BLOCK, n_an=1)
    base = {
        "p_class": belief.p_class.at[0, 0].set(0.6),
        "status": jnp.zeros_like(belief.status),
    }
    base.update(leaves)
    belief = dataclasses.replace(belief, **base)
    view = types.SimpleNamespace(
        belief=belief, tick=jnp.asarray(0), n_dec=jnp.asarray(1)
    )
    effects, new, _ = policy.decide(_state(), view, study, jax.random.key(0))
    return effects, new


@pytest.mark.medium(reason=POLICY_REASON)
def test_the_reserve_blocks_a_search_that_would_spend_what_an_open_track_needs():
    """A blind search starts only if the time left covers it and the track's needs.

    The track holds one detection, so it needs two more detections and one spectrum at
    the heuristic time plus overhead. Fails if the search ignores the reserve, if the
    reserve is not the sum over remaining detections and spectra, or if a reserve of
    zero changes the old rule.
    """
    study = _built()[1]
    cost = np.asarray(study.channels.cost[:, br.SHARED_RESOURCE])
    model = study.channel_model
    over_det = cost[0] - model.exposures_s[0] / 86_400.0
    over_char = cost[br.N_DETECT] - model.block_days[0]
    reserve = 2 * (15.0 + over_det) + (15.0 + over_char)
    enough = float(reserve + cost[0] + 0.1)
    tight = float(reserve + cost[0] - 0.1)
    effects, _ = _search_case(15.0, enough)
    assert int(effects.kind[0]) == EffectKind.OBSERVE
    assert int(effects.subject[0]) == 1 and int(effects.channel_index[0]) == 0
    effects, _ = _search_case(15.0, tight)
    assert int(effects.kind[0]) == EffectKind.WAIT, "the search spent the reserve"
    effects, _ = _search_case(0.0, tight)
    assert int(effects.kind[0]) == EffectKind.OBSERVE, "a zero reserve is the old rule"


@pytest.mark.medium(reason=POLICY_REASON)
def test_the_block_and_follow_up_thresholds_are_knobs():
    """`block_p` and `followup_p` move the starts; a survey channel defaults to 0.95."""
    nine = {
        "p_block": jnp.zeros((N_STARS, N_SLOTS, N_GRID, br.N_BLOCK)).at[0, 0].set(0.9)
    }
    effects, _ = _decide(**nine)
    assert _chose_block(effects), "0.9 clears the first-summary default of 0.8"
    effects, _ = _decide(policy_kw={"block_p": 0.95}, **nine)
    assert not _chose_block(effects)
    effects, _ = _decide(survey=True, **nine)
    assert not _chose_block(effects), "a survey channel needs 0.95"
    effects, _ = _decide(survey=True, policy_kw={"block_p": 0.85}, **nine)
    assert _chose_block(effects)
    # a one-detection slot with p_det 0.4 is followed up only below the gate of 0.4
    low = {
        "n_det": _at(1),
        "p_det": jnp.zeros((N_STARS, N_SLOTS, N_GRID, br.N_DETECT)).at[0, 0].set(0.4),
    }
    effects, _ = _decide(tick_d=20.0, **low)
    assert not _chose_block(effects) and int(effects.kind[0]) == EffectKind.WAIT
    effects, _ = _decide(tick_d=20.0, policy_kw={"followup_p": 0.4}, **low)
    assert int(effects.kind[0]) == EffectKind.OBSERVE


def test_default_capacities_scale_with_stars_and_horizon():
    """A 150-star, 1825-day study gets room; a small one keeps its fixed values.

    The 150-star values are at least those the earlier 150-star runs needed to finish
    (4096 log rows, 96 counting rows would have been generous, 16384 ids).
    """
    days = br.TICKS_PER_DAY
    big = br.default_capacities(150, 4, n_channels=25, horizon_d=1825.0)
    assert big.n_log >= 4096 and big.rows_per_subject >= 1 + 24 + 4
    assert big.max_sid >= 16384
    assert big.max_tick > 1825 * days
    assert big.n_pstate == br.pstate_size(150, 4)
    small = br.default_capacities(2, 6, n_channels=10, horizon_d=120.0)
    assert (small.n_log, small.rows_per_subject, small.max_sid) == (1024, 32, 4096)
    legacy = br.default_capacities(2, 6, n_channels=10)
    assert (legacy.n_log, legacy.rows_per_subject, legacy.max_sid) == (1024, 32, 4096)
    longer = br.default_capacities(150, 4, n_channels=25, horizon_d=1825.0, n_char=4)
    assert longer.rows_per_subject > big.rows_per_subject
    assert longer.n_pstate == br.pstate_size(150, 4, 4)
    with pytest.raises(ValueError, match="max_tick"):
        br.default_capacities(150, 4, n_channels=25, horizon_d=60_000.0)


@pytest.mark.slow(reason="compiles the chunked refresh driver once")
def test_the_block_log_maps_every_block_row_to_its_slot():
    """A run at n_char 2: attempts, spectra and confirmation agree with the log.

    Fails if a block row is attributed to a slot that was not blocked, if the attempts
    do not add up to the blocks run, or if a slot is confirmed with fewer than n_char
    spectra.
    """
    _, study, _, truths, _, run = build(budget_d=30.0, policy_kw={"n_char": 2})
    carry = run.carry
    assert not bool(np.any(np.asarray(carry.overflow)))
    table = br.outcome_table(run, truths, study)
    slot, star = table.slot, table.star
    assert (slot["block_attempts"].sum(axis=2) == star["n_block"]).all()
    assert star["n_block"].sum() > 0
    assert (slot["n_spectra"] <= slot["block_attempts"]).all()
    assert (slot["block_attempts"] <= 3 * 2).all()
    # the outcome table confirms at the policy's n_char unless told otherwise
    one = br.outcome_table(run, truths, study, n_char=1)
    assert (table.world["confirmed"] <= one.world["confirmed"]).all()
