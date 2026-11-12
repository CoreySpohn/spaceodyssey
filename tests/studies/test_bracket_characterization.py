"""Characterization in the bracket policy: retries, retirement, `n_char`, the reserve.

The decision tests hand the policy a summary and a state and read one decision, so no
study runs; the run test checks the block log against the outcome table.
"""

import dataclasses
import functools
import math
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
    return build(run=False, policy_kw={"retire_after": br.RETIRE_AFTER})


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


def _decide(
    pstate=None, tick_d=0.0, policy_kw=None, survey=None, infinite=(), **leaves
):
    """`(effects, new state)` of one decision on a candidate at slot 0 of star 0."""
    _, study, _, _, _, _ = _built()
    policy = dataclasses.replace(
        study.agenda.policies[0], **{"budget_d": 1000.0, **(policy_kw or {})}
    )
    if len(infinite):
        # channels whose solved time is infinite: an infeasible requirement
        cost = study.channels.cost.at[jnp.asarray(infinite), br.SHARED_RESOURCE].set(
            jnp.inf
        )
        study = dataclasses.replace(
            study, channels=dataclasses.replace(study.channels, cost=cost)
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


def _decide_budget(kw, pstate=None, **leaves):
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
    pstate = _state() if pstate is None else pstate
    effects, new, _ = policy.decide(pstate, view, study, jax.random.key(0))
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
    assert (small.n_log, small.max_sid) == (1024, 4096)
    assert small.rows_per_subject >= 1 + 24 + br.block_log_len(6) + 3
    legacy = br.default_capacities(2, 6, n_channels=10)
    assert (legacy.n_log, legacy.rows_per_subject, legacy.max_sid) == (1024, 32, 4096)
    longer = br.default_capacities(150, 4, n_channels=25, horizon_d=1825.0, n_char=2)
    assert longer.rows_per_subject > big.rows_per_subject
    assert longer.n_pstate == br.pstate_size(150, 4, 2)
    with pytest.raises(ValueError, match="max_tick"):
        br.default_capacities(150, 4, n_channels=25, horizon_d=60_000.0)
    with pytest.raises(ValueError, match="max_tick"):
        br.default_capacities(150, 4, n_channels=25, horizon_d=1825.0, n_char=4)
    # A full 1059-star list over five years fits only on a longer declared tick.
    with pytest.raises(ValueError, match="tick_length_s"):
        br.default_capacities(1059, 4, n_channels=25, horizon_d=1826.0)
    second = br.default_capacities(
        1059, 4, n_channels=25, horizon_d=1826.0, ticks_per_day=86_400
    )
    assert second.max_tick > 1826 * 86_400


@pytest.mark.slow(reason="compiles the chunked refresh driver once")
def test_the_block_log_maps_every_block_row_to_its_slot():
    """A run at n_char 2: attempts, spectra and confirmation agree with the log.

    Fails if a block row is attributed to a slot that was not blocked, if the attempts
    do not add up to the blocks run, or if a slot is confirmed with fewer than n_char
    spectra.
    """
    _, study, _, truths, _, run = build(
        budget_d=30.0, policy_kw={"n_char": 2, "retire_after": 3}
    )
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


def test_capacities_cover_every_visit_block_and_renewal_a_star_can_add():
    """`n_log` and `rows_per_subject` are bounds: the visit cap, the block log, refits.

    Each star holds a catalog row, `max_visits` visits and `block_log_len` blocks; each
    visit, block and forecast renewal adds `REFIT_ROWS` refit products. Fails if the
    log is sized from a typical visit count instead of the policy's cap.
    """
    n, slots, horizon = 150, 4, 1825.0
    for max_visits, n_char, retire in ((24, 1, 3), (24, 2, 3), (40, 1, 1), (8, 1, 3)):
        caps = br.default_capacities(
            n,
            slots,
            n_channels=25,
            horizon_d=horizon,
            n_char=n_char,
            retire_after=retire,
            max_visits=max_visits,
        )
        blocks = br.block_log_len(slots, n_char, retire)
        renewals = math.ceil(horizon / br.RENEWAL_D) + 1
        per_star = 1 + max_visits + blocks
        refits = br.REFIT_ROWS * (max_visits + blocks + renewals)
        assert caps.n_log >= n * (per_star + refits)
        assert caps.rows_per_subject >= per_star
        assert caps.max_sid >= 2 * n * (per_star + refits)
        assert caps.max_tick > horizon * br.TICKS_PER_DAY


@pytest.mark.medium(reason=POLICY_REASON)
def test_a_slot_no_longer_mode_serves_is_retired_and_holds_no_reserve():
    """A failure that no longer mode can follow retires the slot.

    Fails if the stranded slot keeps counting as an open track, so that the reserve
    holds its remaining detections and spectrum back from blind searches.
    """
    study = _built()[1]
    cost = np.asarray(study.channels.cost[:, br.SHARED_RESOURCE])
    model = study.channel_model
    over_det = cost[0] - model.exposures_s[0] / 86_400.0
    over_char = cost[br.N_DETECT] - model.block_days[0]
    reserve = 2 * (15.0 + over_det) + (15.0 + over_char)
    tight = float(reserve + cost[0] - 0.1)
    p_new = jnp.zeros((N_STARS, N_GRID, br.N_DETECT)).at[1, :, 0].set(0.5)
    base = {
        "n_det": _at(1),
        "p_det": jnp.zeros((N_STARS, N_SLOTS, N_GRID, br.N_DETECT)),
        "p_new": p_new,
        "n_blocks": _read(1),
    }
    top = br.N_BLOCK - 1
    served = jnp.zeros((N_STARS, N_SLOTS, N_GRID, br.N_BLOCK)).at[0, 0].set(1.0)
    short = served.at[0, 0, :, 1:].set(0.0)
    cases = {
        # mode 0 failed with every longer mode forecast low
        "forecast": (_state(att=1, last_mode=0, last_ord=1, n_blk=1), short, True),
        "longest": (_state(att=1, last_mode=top, last_ord=1, n_blk=1), served, True),
        # a longer mode still serves the slot: it stays open and holds its reserve
        "open": (_state(att=1, last_mode=0, last_ord=1, n_blk=1), served, False),
    }
    for name, (pstate, p_block, retired) in cases.items():
        leaves = {**base, "p_block": p_block}
        kw = {"reserve_d": 15.0, "budget_d": tight}
        effects, _ = _decide_budget(kw, pstate=pstate, **leaves)
        searched = (
            int(effects.kind[0]) == EffectKind.OBSERVE
            and int(effects.subject[0]) == 1
            and int(effects.channel_index[0]) == 0
        )
        assert searched == retired, name


@pytest.mark.medium(reason=POLICY_REASON)
def test_a_block_the_budget_cannot_cover_is_not_run_as_a_visit():
    """Budget gone mid-retry: no stand-in visit, no visit counted, the policy waits.

    Fails if the unaffordable block runs the cheapest visit (and would again at every
    decision until the budget is gone).
    """
    study = _built()[1]
    cost = np.asarray(study.channels.cost[:, br.SHARED_RESOURCE])
    failed = _state(att=1, last_mode=1, last_ord=1, n_blk=1)
    retry = br.N_DETECT + 2
    between = float((cost[retry] + cost[0]) / 2)
    assert cost[0] <= between < cost[retry]
    effects, new = _decide(failed, policy_kw={"budget_d": between}, n_blocks=_read(1))
    assert int(effects.kind[0]) == EffectKind.WAIT
    assert new[1] == 0 and new[N_STARS * (3 + N_SLOTS)] == 0, "a visit was charged"
    # the same state with room runs the retry
    effects, _ = _decide(failed, n_blocks=_read(1))
    assert int(effects.channel_index[0]) == retry
    # nothing fits at all: still no visit
    effects, new = _decide(
        failed, policy_kw={"budget_d": float(cost[0]) / 2}, n_blocks=_read(1)
    )
    assert int(effects.kind[0]) == EffectKind.WAIT and new[1] == 0


@pytest.mark.medium(reason=POLICY_REASON)
def test_n_char_two_retires_after_failures_that_follow_a_success():
    """After one delivered spectrum, three failures in a row retire; two do not.

    The success clears the failure count, so the run of failures starts from zero.
    Fails if the success is not forgiven or if the slot is retired early.
    """
    kw = {"n_char": 2}
    one = {"revealed": _at(CLASS_ONE_CODE), "n_spectra": _at(1), "n_blocks": _read(4)}
    # blocks: success, fail, fail: the third failure is the next one read
    two_failed = _state(
        n_char=2, att=3, cons=1, seen=1, last_mode=1, last_ord=3, n_blk=3
    )
    effects, _ = _decide(two_failed, policy_kw=kw, **{**one, "n_blocks": _read(3)})
    assert _chose_block(effects), "two failures after a success are not retirement"
    three = _state(n_char=2, att=4, cons=2, seen=1, last_mode=1, last_ord=4, n_blk=4)
    effects, _ = _decide(three, policy_kw=kw, **one)
    assert not _chose_block(effects), "three failures after a success retire the slot"


@pytest.mark.medium(reason=POLICY_REASON)
def test_a_mode_of_infinite_time_is_never_chosen():
    """Channels with an infinite cost (an infeasible requirement) are never picked.

    Fails if a block or a search is started on a mode that cannot finish.
    """
    blocks = range(br.N_DETECT, br.N_DETECT + br.N_BLOCK)
    effects, _ = _decide(infinite=tuple(blocks))
    assert not _chose_block(effects) and int(effects.kind[0]) == EffectKind.WAIT
    p_new = jnp.zeros((N_STARS, N_GRID, br.N_DETECT)).at[1, :, 0].set(0.5)
    leaves = {
        "p_new": p_new,
        "p_det": jnp.zeros((N_STARS, N_SLOTS, N_GRID, br.N_DETECT)),
        "p_block": jnp.zeros((N_STARS, N_SLOTS, N_GRID, br.N_BLOCK)),
    }
    effects, _ = _decide(**leaves)
    assert int(effects.kind[0]) == EffectKind.OBSERVE, "the finite search runs"
    assert int(effects.subject[0]) == 1 and int(effects.channel_index[0]) == 0
    effects, _ = _decide(infinite=(0,), **leaves)
    assert int(effects.kind[0]) == EffectKind.WAIT
