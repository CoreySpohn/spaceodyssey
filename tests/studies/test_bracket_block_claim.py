"""A delivered block decides the claim: nomination, reveal, rejection, outcomes."""

import dataclasses
import functools
import types

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey.adapters import jaxedith as jx
from spaceodyssey.adapters.photomancy import N_GRID, SLOT_KEPT, SLOT_OK, DecisionSummary
from spaceodyssey.core import index_world, replay_run
from spaceodyssey.core.enums import Audience
from spaceodyssey.studies import bracket as br

from .bracket_helpers import N_SLOTS, build

# Enough telescope days that the fixture's first candidate gets a block.
BUDGET_D = 30.0
CLASS_ONE_CODE = br.CLASS_ONE + br.REVEAL_OFFSET


def test_a_blocks_class_column_is_the_only_one_an_observer_may_read():
    """`revealed_class` is open to the observer and policy; the rest is scorer-only."""
    kind = jx.block_outcome_kind(N_SLOTS)
    reader = int(Audience.OBSERVER) | int(Audience.POLICY)
    for column in kind.columns:
        released = (int(column.audience) & reader) == reader
        assert released == column.name.startswith("revealed_class"), column.name
        assert int(column.audience) & int(Audience.SCORER), column.name
    cols = jx.reveal_columns(N_SLOTS)
    names = jx.block_columns(N_SLOTS)
    assert [names[c] for c in cols] == [f"revealed_class_{s}" for s in range(N_SLOTS)]


@functools.cache
def _oracle(block_snr_required=None):
    return build(budget_d=BUDGET_D, block_snr_required=block_snr_required)


def _block_rows(run, truths, study):
    """`(world, star, payload)` of every valid block row of the run."""
    log = run.carry.log
    kind, valid = np.asarray(log.kind), np.asarray(log.valid)
    out = []
    for w in range(kind.shape[0]):
        for r in np.flatnonzero((kind[w] == br.KIND_BLOCK) & valid[w]):
            out.append(
                (w, int(np.asarray(log.subject)[w, r]), np.asarray(log.payload)[w, r])
            )
    return out


@pytest.mark.slow(reason="compiles the chunked refresh driver once")
@pytest.mark.parametrize("required", [None, 5.0])
def test_a_confirmed_slot_is_class_one_and_only_a_delivered_block_reveals(required):
    """Over every world: no false claim, and no reveal without a delivered block.

    Fails if the realization, the belief or the outcome table reads another block
    requirement than the channel's `block_snr_required` (checked at its default and
    at 5, the characterization requirement of a lower resolving power).
    """
    _, study, _, truths, _, run = _oracle(required)
    threshold = float(study.channel_model.block_snr_required)
    assert threshold == (jx.BLOCK_SNR_REQUIRED if required is None else required)
    table = br.outcome_table(run, truths, study)
    present, klass = np.asarray(truths.present), np.asarray(truths.klass)
    slot = table.slot
    assert (table.world["false_claims"] == 0).all()
    assert not (slot["claimed"] & ~(present & (klass == 1))).any()
    assert (table.world["confirmed"] == slot["claimed"].sum(axis=(1, 2))).all()

    # What each block row delivered, slot by slot, from the true signal-to-noise.
    names = jx.block_columns(N_SLOTS)
    snr = [jx.column_index(names, f"block_snr_{k}") for k in range(N_SLOTS)]
    rev = jx.reveal_columns(N_SLOTS)
    delivered = np.zeros(present.shape, bool)
    rows = _block_rows(run, truths, study)
    assert rows, "no block was started"
    for w, s, payload in rows:
        met = (payload[snr] >= threshold) & present[w, s]
        expected = np.where(met, klass[w, s] + br.REVEAL_OFFSET, 0)
        np.testing.assert_array_equal(payload[list(rev)], expected)
        delivered[w, s] |= met
    revealed = slot["revealed"] >= 0
    assert not (revealed & ~delivered).any(), "a reveal without a delivered block"
    assert (slot["revealed"][revealed] == klass[revealed]).all()
    # A delivered block is a block on the slot: wasted blocks are never confirmed.
    assert (slot["block_delivered"] <= slot["claimed"]).all()


def _install():
    zeros = DecisionSummary.zeros(2, 3, N_GRID, 1, 1, n_an=1)
    observer = br.SummaryObserver(prior=zeros)
    return eqx.filter_jit(lambda old, new: observer.install(old, new, 0)), zeros


def test_a_reveal_overrides_the_class_probability_even_on_a_kept_slot():
    """Class 1 reads 1, any other class reads 0, and a failed slot still carries it."""
    install, zeros = _install()
    old = dataclasses.replace(zeros, p_class=jnp.full((2, 3), 0.4))
    new = dataclasses.replace(
        zeros,
        p_class=jnp.full((2, 3), 0.3),
        status=jnp.asarray([[SLOT_OK, SLOT_OK, SLOT_KEPT[0]], [0, 0, 0]]),
        revealed=jnp.asarray(
            [[CLASS_ONE_CODE, br.REVEAL_OFFSET, CLASS_ONE_CODE], [0] * 3]
        ),
    )
    out = install(old, new)
    assert np.allclose(out.p_class[0], [1.0, 0.0, 1.0])
    assert np.array_equal(out.revealed[0], new.revealed[0])
    assert np.array_equal(np.asarray(br.confirmed_slots(out))[0], [1, 0, 1])


@functools.cache
def _study():
    return build(run=False)


def _decide(build_out, tick_d=0.0, **leaves):
    """One decision on a summary whose slot 0 of star 0 holds `leaves`."""
    _, study, _, _, _, _ = build_out
    caps = study.caps
    policy = dataclasses.replace(study.agenda.policies[0], budget_d=1000.0)
    n, slots = caps.n_subjects, policy.n_slots
    belief = DecisionSummary.zeros(n, slots, N_GRID, br.N_DETECT, br.N_BLOCK, n_an=1)
    base = {
        "n_det": belief.n_det.at[0, 0].set(3),
        "p_class": belief.p_class.at[0, 0].set(0.6),
        "status": jnp.zeros_like(belief.status),
        "p_block": belief.p_block.at[0, 0].set(1.0),
        "p_det": belief.p_det.at[0, 0].set(1.0),
    }
    base.update(leaves)
    belief = dataclasses.replace(belief, **base)
    pstate = jnp.zeros(caps.n_pstate).at[n * (3 + slots) + 1].set(float(n))
    view = types.SimpleNamespace(
        belief=belief,
        tick=jnp.asarray(round(tick_d * 86_400_000)),
        n_dec=jnp.asarray(1),
    )
    effects, new, _ = policy.decide(pstate, view, study, jax.random.key(0))
    stars = np.asarray(new)[: n * (3 + slots)].reshape(n, 3 + slots)
    return int(effects.channel_index[0]), stars


# The policy tests compile the whole bracket policy once per file (about 17 s on the
# hosted runner), over the unit budget.
POLICY_REASON = "compiles the bracket policy decision program"


def _slots(built, value, dtype=jnp.int64):
    """A ``[n_stars, n_slots]`` array that is `value` at slot 0 of star 0."""
    shape = (built[1].caps.n_subjects, N_SLOTS)
    return jnp.zeros(shape, dtype).at[0, 0].set(value)


@pytest.mark.medium(reason=POLICY_REASON)
def test_a_candidate_is_blocked_and_a_revealed_slot_is_not():
    """Rule 1 acts on a candidate; a revealed slot (class 1 or not) gets no block."""
    built = _study()
    channel, stars = _decide(built)
    assert channel >= br.N_DETECT and stars[0, 3] == 1.0, "the candidate is blocked"
    for code in (CLASS_ONE_CODE, br.REVEAL_OFFSET):
        channel, stars = _decide(built, revealed=_slots(built, code))
        assert stars[:, 3:].sum() == 0.0, f"code {code} was blocked"
        assert channel < br.N_DETECT


@pytest.mark.medium(reason=POLICY_REASON)
def test_a_revealed_slot_is_not_followed_up():
    """A one-detection slot is followed up; once a block revealed it, it is not."""
    built = _study()
    one = {"n_det": _slots(built, 1)}
    _, stars = _decide(built, tick_d=20.0, **one)
    assert stars[0, 1] == 1.0, "the unrevealed slot is followed up"
    for code, p_class in ((br.REVEAL_OFFSET, 0.0), (CLASS_ONE_CODE, 1.0)):
        _, stars = _decide(
            built,
            tick_d=20.0,
            p_class=_slots(built, p_class, jnp.float64),
            revealed=_slots(built, code),
            **one,
        )
        assert stars[:, 1].sum() == 0.0, f"code {code} was followed up"


def _p_block(built, *by_mode):
    """`p_block` zero but for slot 0 of star 0: one node curve per block mode."""
    n = built[1].caps.n_subjects
    out = jnp.zeros((n, N_SLOTS, N_GRID, br.N_BLOCK))
    for mode, curve in enumerate(by_mode):
        out = out.at[0, 0, :, mode].set(jnp.asarray(curve, jnp.float64))
    return out


@pytest.mark.medium(reason=POLICY_REASON)
def test_a_candidate_takes_the_shortest_block_mode_that_can_succeed():
    """Each candidate is blocked at its own shortest mode that is good now and here.

    Fails if a mode that cannot reach the requirement (p_block 0 at every node) is
    chosen because a fixed per-star mode said so, if a longer mode is skipped when the
    shorter one is better at a later node only, or if a candidate no mode serves is
    blocked anyway.
    """
    built = _study()
    flat, none = np.ones(N_GRID), np.zeros(N_GRID)
    later = np.full(N_GRID, 0.85)
    later[5] = 1.0
    cases = [
        ((flat, flat, flat), 0),
        ((none, flat, flat), 1),
        ((none, none, flat), 2),
        ((later, flat, flat), 1),
    ]
    for curves, mode in cases:
        channel, stars = _decide(built, p_block=_p_block(built, *curves))
        assert channel == br.N_DETECT + mode, curves
        assert stars[0, 3] == 1.0
    channel, stars = _decide(built, p_block=_p_block(built, none, none, none))
    assert stars[:, 3:].sum() == 0.0, "a candidate no mode serves was blocked"
    assert channel < br.N_DETECT


@pytest.mark.slow(reason="a refresh study past its forecast window, replayed")
def test_a_star_past_its_forecast_window_is_refit_without_new_rows():
    """The summary is renewed before its clock passes the last forecast node.

    Fails if a star's forecast freezes at its last node once the clock is past the
    window (the policy then never sees the node a block would succeed at), or if the
    wait-time renewal is not reproduced by `replay_run`.
    """
    horizon_d = 2.0 * br.N_NODES * br.NODE_SPACING_D
    _, study, executor, _, roots, run = build(horizon_d=horizon_d, budget_d=BUDGET_D)
    last = np.asarray(run.carry.pbelief.ref_tick)
    now = np.asarray(run.carry.tick)
    window = (br.N_NODES - 1) * br.NODE_SPACING_D * br.TICKS_PER_DAY
    assert (now[:, None] - last <= window).all(), "a forecast window ran out"
    for w in range(roots.shape[0]):
        replay_run(
            study,
            roots[w],
            index_world(run.carry, w),
            index_world(run.trace, w),
            refit=executor.boundary,
        )


def test_the_observer_expires_a_summary_at_its_last_node():
    """`expired` is true from the last node on and false before it.

    Fails if the renewal threshold drifts from the policy's current-node rule
    (`g_cur` reaching the last node).
    """
    zeros = DecisionSummary.zeros(2, 3, N_GRID, 1, 1, n_an=1)
    zeros = dataclasses.replace(zeros, ref_tick=jnp.asarray([0, 0]))
    observer = br.SummaryObserver(prior=zeros)
    caps = types.SimpleNamespace(n_subjects=2)
    day = br.TICKS_PER_DAY
    spacing = br.NODE_SPACING_D
    before = (br.N_NODES - 1.6) * spacing * day
    at_last = (br.N_NODES - 1.4) * spacing * day
    assert not observer.expired(zeros, jnp.asarray(round(before)), caps).any()
    assert observer.expired(zeros, jnp.asarray(round(at_last)), caps).all()
