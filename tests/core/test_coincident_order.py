"""A channel's coincident order ranks a fault against a settlement at one tick."""

import equinox as eqx
import jax.numpy as jnp
import pytest

from spaceodyssey import core
from spaceodyssey.core.enums import (
    CoincidentOrder,
    EventKind,
    RecordStatus,
    Stage,
)
from spaceodyssey.studies import toy as kinds
from tests.core.helpers import (
    CAPS,
    admit_j,
    advance_j,
    observe_effect,
    one_world,
)

TRUTH, ROOT = one_world()
ORDERS = [CoincidentOrder.SETTLE_FIRST, CoincidentOrder.FAULT_FIRST]


def study_with(order):
    base = kinds.survey_1()
    return eqx.tree_at(
        lambda s: s.channels.coincident_order,
        base,
        jnp.full(CAPS.n_channels, order, jnp.int64),
    )


def with_pending_settlement(order):
    """A study and carry with one observation of lane 0 queued to settle at 110."""
    study = study_with(order)
    c = eqx.filter_jit(core.init_carry)(study, TRUTH)
    c, _codes = admit_j(c, observe_effect(0, 10), study)
    return study, c


def add_fault(c, study, *, at=110):
    return core.enqueue(
        c,
        True,
        kind=EventKind.FAULT,
        tick=at,
        stage=Stage.SETTLE,
        lane=0,
        channel_index=0,
        channel=10,
        subject=0,
        gen=c.ledger.lane_gen[0],
        tie=core.fault_tie(study.channels, 0),
        caps=CAPS,
    )


def pop_kinds(c, n):
    out = []
    for _ in range(n):
        c, ev, _has = core.select_next(c, CAPS)
        out.append(int(ev.kind))
    return out


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("fault_queued_first", [True, False])
def test_tie_break_follows_the_channel_not_the_queue_order(order, fault_queued_first):
    study = study_with(order)
    c = core.empty_carry(CAPS)
    done = dict(kind=EventKind.DONE, tick=10, stage=Stage.SETTLE, lane=0, caps=CAPS)
    fault = dict(
        kind=EventKind.FAULT,
        tick=10,
        stage=Stage.SETTLE,
        lane=0,
        tie=core.fault_tie(study.channels, 0),
        caps=CAPS,
    )
    rows = [fault, done] if fault_queued_first else [done, fault]
    for row in rows:
        c = core.enqueue(c, True, **row)
    first = EventKind.FAULT if order == CoincidentOrder.FAULT_FIRST else EventKind.DONE
    assert pop_kinds(c, 2)[0] == first


def test_tie_break_comes_after_stage():
    study = study_with(CoincidentOrder.FAULT_FIRST)
    c = core.empty_carry(CAPS)
    c = core.enqueue(
        c,
        True,
        kind=EventKind.FAULT,
        tick=10,
        stage=Stage.RELEASE,
        tie=core.fault_tie(study.channels, 0),
        caps=CAPS,
    )
    c = core.enqueue(
        c, True, kind=EventKind.DONE, tick=10, stage=Stage.SETTLE, caps=CAPS
    )
    assert pop_kinds(c, 2) == [EventKind.DONE, EventKind.FAULT]


@pytest.mark.medium(reason="over 3 s cold here, about 5 s on the runner")
def test_fault_first_fails_the_acquisition_and_the_settlement_goes_stale():
    study, c = with_pending_settlement(CoincidentOrder.FAULT_FIRST)
    c = add_fault(c, study)
    for _ in range(3):
        c, _, _ = advance_j(c, ROOT, study, TRUTH)
    assert int(c.n_log) == 1
    assert int(c.log.status[0]) == RecordStatus.FAILED
    assert not bool(c.double_settle)
    # charged in full: nothing comes back
    assert float(c.ledger.balance[0, 0]) == pytest.approx(9.0)


@pytest.mark.medium(reason="over 3 s cold here, about 5 s on the runner")
def test_settle_first_completes_and_the_fault_finds_nothing_to_fail():
    study, c = with_pending_settlement(CoincidentOrder.SETTLE_FIRST)
    c = add_fault(c, study)
    for _ in range(3):
        c, _, _ = advance_j(c, ROOT, study, TRUTH)
    assert int(c.n_log) == 1
    assert int(c.log.status[0]) == RecordStatus.COMPLETED
    assert float(c.ledger.balance[0, 0]) == pytest.approx(9.0)


def test_coincident_order_is_a_required_column():
    base = kinds.survey_1().channels
    fields = {
        f: getattr(base, f)
        for f in base.__dataclass_fields__
        if f != "coincident_order"
    }
    with pytest.raises(TypeError):
        type(base)(**fields)
