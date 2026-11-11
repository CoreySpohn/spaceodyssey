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
def _oracle():
    return build(budget_d=BUDGET_D)


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
def test_a_confirmed_slot_is_class_one_and_only_a_delivered_block_reveals():
    """Over every world: no false claim, and no reveal without a delivered block."""
    _, study, _, truths, _, run = _oracle()
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
        met = (payload[snr] >= jx.BLOCK_SNR_REQUIRED) & present[w, s]
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


def _slots(built, value, dtype=jnp.int64):
    """A ``[n_stars, n_slots]`` array that is `value` at slot 0 of star 0."""
    shape = (built[1].caps.n_subjects, N_SLOTS)
    return jnp.zeros(shape, dtype).at[0, 0].set(value)


def test_a_candidate_is_blocked_and_a_revealed_slot_is_not():
    """Rule 1 acts on a candidate; a revealed slot (class 1 or not) gets no block."""
    built = _study()
    channel, stars = _decide(built)
    assert channel >= br.N_DETECT and stars[0, 3] == 1.0, "the candidate is blocked"
    for code in (CLASS_ONE_CODE, br.REVEAL_OFFSET):
        channel, stars = _decide(built, revealed=_slots(built, code))
        assert stars[:, 3:].sum() == 0.0, f"code {code} was blocked"
        assert channel < br.N_DETECT


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
