"""Tests for the event queue: ordering keys, micro placement, overflow, selection."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from spaceodyssey import core as sch
from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.enums import BIG, N_STAGES, Audience, EventKind, Stage

CAPS = Capacities()


def order_key_np(tick, stage, micro, sid, caps):
    """Python-int twin of `order_key` (no retrace per hypothesis example)."""
    return ((tick * N_STAGES + stage) * caps.max_micro + micro) * caps.max_sid + sid


def carry_at(caps=CAPS, **fields):
    c = sch.empty_carry(caps)
    return eqx.tree_at(
        lambda t: tuple(getattr(t, k) for k in fields),
        c,
        tuple(jnp.asarray(v, dtype=getattr(c, k).dtype) for k, v in fields.items()),
    )


def enqueue_j(c, ok, caps=CAPS, **kw):
    return eqx.filter_jit(lambda c_: sch.enqueue(c_, ok, caps=caps, **kw))(c)


def select_next_j(c, caps=CAPS):
    return eqx.filter_jit(lambda c_: sch.select_next(c_, caps))(c)


def row(c, i=0):
    return jax.tree_util.tree_map(lambda a: a[i], c.ev)


def row_of(c, kind):
    i = int(jnp.argmax(c.ev.valid & (c.ev.kind == kind)))
    return row(c, i)


def fill_queue(caps):
    c = sch.empty_carry(caps)
    for i in range(caps.n_events):
        c = enqueue_j(
            c, True, caps, kind=EventKind.WAIT, tick=i + 1, stage=Stage.RELEASE
        )
    return c


@given(
    tick=st.integers(0, Capacities().max_tick),
    stage=st.integers(0, N_STAGES - 1),
    micro=st.integers(0, Capacities().max_micro - 1),
    sid=st.integers(0, Capacities().max_sid - 1),
)
@settings(max_examples=50, deadline=None, derandomize=True)
def test_order_key_is_lexicographic(tick, stage, micro, sid):
    caps = Capacities()
    k = order_key_np(tick, stage, micro, sid, caps)
    assert k < BIG
    if sid + 1 < caps.max_sid:
        assert k < order_key_np(tick, stage, micro, sid + 1, caps)
    if micro + 1 < caps.max_micro:
        assert k < order_key_np(tick, stage, micro + 1, 0, caps)
    if stage + 1 < N_STAGES:
        assert k < order_key_np(tick, stage + 1, 0, 0, caps)
    if tick < caps.max_tick:
        assert k < order_key_np(tick + 1, 0, 0, 0, caps)


def test_order_key_jax_matches_python_twin():
    caps = Capacities()
    got = int(sch.order_key(jnp.asarray(123456789), 3, 5, 77, caps))
    assert got == order_key_np(123456789, 3, 5, 77, caps)


def test_caps_rejects_bad_guard():
    with pytest.raises(ValueError):
        Capacities(micro_guard=8, max_micro=8)


def test_caps_rejects_more_solves_than_subjects():
    with pytest.raises(ValueError):
        Capacities(n_subjects=2, n_an=4)


def test_caps_rejects_key_overflow():
    with pytest.raises(ValueError):
        Capacities(max_sid=2**55)


def test_caps_derived_values():
    caps = Capacities()
    assert caps.n_balance_rows == caps.n_assets + caps.n_programs + 1
    assert caps.max_tick > 10**12


def test_micro_placement_under_jit():
    c = carry_at(tick=10, stage=Stage.RELEASE, micro=2)
    same = enqueue_j(c, True, kind=EventKind.WAIT, tick=10, stage=Stage.RELEASE)
    later = enqueue_j(c, True, kind=EventKind.STOP, tick=10, stage=Stage.ADMIT)
    earlier = enqueue_j(c, True, kind=EventKind.DONE, tick=10, stage=Stage.SETTLE)
    assert int(row_of(same, EventKind.WAIT).micro) == 3
    assert int(row_of(later, EventKind.STOP).micro) == 0
    r = row_of(earlier, EventKind.DONE)
    assert int(r.stage) == Stage.RELEASE and int(r.micro) == 3
    assert bool(earlier.violation) and not bool(same.violation)


def test_later_tick_gets_micro_zero():
    c = carry_at(tick=10, stage=Stage.ADMIT, micro=5)
    later = enqueue_j(c, True, kind=EventKind.WAIT, tick=11, stage=Stage.SETTLE)
    assert int(row_of(later, EventKind.WAIT).micro) == 0 and not bool(later.violation)


def test_enqueue_not_ok_changes_nothing():
    c = carry_at(tick=1)
    c2 = enqueue_j(c, False, kind=EventKind.WAIT, tick=5, stage=Stage.RELEASE)
    assert not bool(jnp.any(c2.ev.valid)) and int(c2.next_sid) == 0
    assert not bool(c2.overflow)


def test_enqueue_full_sets_overflow_keeps_rows():
    caps = Capacities(n_events=4)
    c = fill_queue(caps)
    assert not bool(c.overflow)
    c2 = enqueue_j(c, True, caps, kind=EventKind.WAIT, tick=99, stage=Stage.RELEASE)
    assert bool(c2.overflow) and jnp.array_equal(c2.ev.tick, c.ev.tick)
    assert int(c2.next_sid) == int(c.next_sid)


def test_sid_exhaustion_sets_overflow():
    c = carry_at(next_sid=Capacities().max_sid)
    c2 = enqueue_j(c, True, kind=EventKind.WAIT, tick=5, stage=Stage.RELEASE)
    assert bool(c2.overflow) and not bool(jnp.any(c2.ev.valid))


def test_sid_increments_per_write():
    c = enqueue_j(carry_at(), True, kind=EventKind.WAIT, tick=5, stage=Stage.RELEASE)
    c = enqueue_j(c, True, kind=EventKind.WAIT, tick=6, stage=Stage.RELEASE)
    assert int(c.next_sid) == 2
    assert sorted(int(s) for s in c.ev.sid[c.ev.valid]) == [0, 1]


def test_select_next_done_before_release_same_tick():
    c = enqueue_j(carry_at(), True, kind=EventKind.RELEASE, tick=7, stage=Stage.RELEASE)
    c = enqueue_j(c, True, kind=EventKind.DONE, tick=7, stage=Stage.SETTLE)
    c, ev, has = select_next_j(c)
    assert bool(has) and int(ev.kind) == EventKind.DONE and int(c.stage) == Stage.SETTLE
    assert int(c.tick) == 7


def test_select_next_clears_slot_and_orders_by_sid_on_ties():
    c = carry_at()
    for _ in range(3):
        c = enqueue_j(c, True, kind=EventKind.DONE, tick=7, stage=Stage.SETTLE)
    sids = []
    for _ in range(3):
        c, ev, _has = select_next_j(c)
        sids.append(int(ev.sid))
    assert sids == [0, 1, 2] and not bool(jnp.any(c.ev.valid))


def test_select_next_empty_queue_reports_no_row():
    c = carry_at(tick=4, stage=Stage.RELEASE, micro=1)
    c2, ev, has = select_next_j(c)
    assert not bool(has) and not bool(ev.valid)
    assert int(c2.tick) == 4 and int(c2.micro) == 1 and not bool(c2.stalled)


def test_stall_guard_trips_at_micro_guard():
    c = carry_at(tick=3, stage=Stage.ADMIT, micro=0)
    for _i in range(Capacities().micro_guard):
        assert not bool(c.stalled)
        c = enqueue_j(c, True, kind=EventKind.WAIT, tick=3, stage=Stage.ADMIT)
        c, _, _ = select_next_j(c)
    assert bool(c.stalled)


def test_where_tree_selects_per_leaf():
    a, b = carry_at(tick=1), carry_at(tick=2)
    assert int(sch.where_tree(jnp.asarray(True), a, b).tick) == 1
    assert int(sch.where_tree(jnp.asarray(False), a, b).tick) == 2


def test_rec_none_is_invalid_and_matches_log_columns():
    r = sch.record_none(CAPS)
    assert not bool(r.valid) and r.cost.shape == (CAPS.n_resources,)
    assert r.payload.shape == (CAPS.payload_width,)
    assert set(sch.RECORD_INT_FIELDS) == set(sch.LOG_INT_FIELDS)


# ---- records, log, view, comparison ------------------------------------


def some_rec(caps=CAPS, value0=1.0, **kw):
    r = sch.record_none(caps)
    r = dataclasses.replace(
        r, valid=jnp.asarray(True), payload=r.payload.at[0].set(value0), **kw
    )
    return r


def write_log_j(c, rec, caps=CAPS):
    return eqx.filter_jit(lambda c_, r_: sch.write_log(c_, r_, caps))(c, rec)


def carry_with_log_full(caps):
    c = sch.empty_carry(caps)
    for i in range(caps.n_log):
        c, _ = write_log_j(c, some_rec(caps, value0=float(i + 1)), caps)
    return c


def log_with_rows(rows, caps=CAPS):
    """Carry whose log holds rows [(kind, payload)].

    Carry whose log holds rows [(kind, payload)]: 'pub' rows are released, 'priv'
    are not.
    """
    c = sch.empty_carry(caps)
    for kind, v in rows:
        c, slot = write_log_j(c, some_rec(caps, value0=float(v)), caps)
        released_tick = 5 if kind == "pub" else -1
        c = eqx.tree_at(
            lambda x: x.log.released_tick,
            c,
            c.log.released_tick.at[slot].set(released_tick),
        )
    return c


def compact_view_j(c, audience, caps=CAPS):
    return eqx.filter_jit(lambda c_: sch.snapshot_view(c_, audience, caps))(c)


def test_write_log_full_sets_overflow_and_skips():
    caps = Capacities(n_log=2)
    c = carry_with_log_full(caps)
    assert not bool(c.overflow)
    c2, slot = write_log_j(c, some_rec(caps, value0=99.0), caps)
    assert bool(c2.overflow) and int(c2.n_log) == 2
    assert jnp.array_equal(c2.log.payload, c.log.payload) and int(slot) == 1


def test_write_log_invalid_rec_writes_nothing_and_is_not_overflow():
    c = sch.empty_carry(CAPS)
    c2, _ = write_log_j(c, sch.record_none(CAPS))
    assert int(c2.n_log) == 0 and not bool(c2.overflow)


def test_write_log_stores_every_column():
    c, slot = write_log_j(
        sch.empty_carry(CAPS),
        some_rec(subject=3, acquisition_index=2, acquisition_id=9),
    )
    assert int(c.log.subject[slot]) == 3 and int(c.log.acquisition_index[slot]) == 2
    assert int(c.log.acquisition_id[slot]) == 9 and bool(c.log.valid[slot])


def test_compact_view_private_rows_do_not_shift_public():
    a = log_with_rows([("pub", 0), ("pub", 1)])
    b = log_with_rows([("priv", 9), ("pub", 0), ("pub", 1)])
    va, vb = compact_view_j(a, Audience.POLICY), compact_view_j(b, Audience.POLICY)
    assert jnp.array_equal(va.log.payload[:2], vb.log.payload[:2])
    assert int(va.n_rows) == int(vb.n_rows) == 2
    assert bool(jnp.all(vb.log.payload[2:] == 0))
    assert bool(jnp.all(vb.log.valid[2:] == False))  # noqa: E712


def test_tree_equal_report_detects_one_leaf():
    c = sch.empty_carry(CAPS)
    d = eqx.tree_at(lambda x: x.belief.mean, c, c.belief.mean.at[0].add(1e-11))
    r = sch.tree_equal_report(c, d)
    assert r["structure_equal"] and r["max_abs"] >= 1e-11 and r["max_int_diff"] == 0
    assert sch.tree_equal_report(c, c)["max_abs"] == 0


def test_tree_equal_report_counts_integer_and_bool_differences():
    c = sch.empty_carry(CAPS)
    d = eqx.tree_at(lambda x: x.n_dec, c, c.n_dec + 3)
    assert sch.tree_equal_report(c, d)["max_int_diff"] == 3
    e = eqx.tree_at(lambda x: x.done, c, jnp.asarray(True))
    assert sch.tree_equal_report(c, e)["max_int_diff"] == 1


def test_tree_equal_report_flags_structure_and_shape():
    assert not sch.tree_equal_report((1, 2), (1,))["structure_equal"]
    r = sch.tree_equal_report(jnp.zeros(3), jnp.zeros(4))
    assert r["structure_equal"] and not r["shapes_equal"]


def test_rec_value_layout():
    r = sch.record_none(Capacities())
    assert (
        r.payload.shape == (4,)
        and r.cost.shape == (3,)
        and r.payload.dtype == jnp.float64
    )


def test_compact_view_hides_internal_settlement_ids():
    c = log_with_rows([("pub", 0), ("pub", 1)])
    c = eqx.tree_at(
        lambda x: x.log.acquisition_id, c, c.log.acquisition_id.at[0].set(77)
    )
    assert bool(jnp.all(compact_view_j(c, Audience.POLICY).log.acquisition_id == 0))
