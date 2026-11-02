"""A drain closes exactly as a close does below campaign scope."""

import dataclasses

import jax.numpy as jnp
import pytest

from spaceodyssey import core
from spaceodyssey.core.enums import Disposition, EventKind, Scope
from tests.core.helpers import after_decision_1_bundle, handle_event, one_world

TRUTH, ROOT = one_world()


def live_state(c):
    """Every leaf of the carry, with the dead rows of the event queue zeroed."""
    live = c.ev.valid
    ev = tuple(
        jnp.where(live, getattr(c.ev, f.name), 0) for f in dataclasses.fields(c.ev)
    )
    rest = tuple(getattr(c, f.name) for f in dataclasses.fields(c) if f.name != "ev")
    return ev, rest


def stopped(scope, disposition):
    study, c = after_decision_1_bundle()
    return handle_event(
        c,
        study,
        TRUTH,
        ROOT,
        EventKind.STOP,
        lane=0,
        scope=scope,
        disposition=disposition,
    )


@pytest.mark.parametrize("scope", [Scope.LANE, Scope.ASSET, Scope.PROGRAM])
def test_stop_equivalence_below_campaign(scope):
    close, drain = stopped(scope, Disposition.CLOSE), stopped(scope, Disposition.DRAIN)
    r = core.tree_equal_report(live_state(close), live_state(drain))
    assert r["structure_equal"] and r["max_int_diff"] == 0 and r["max_abs"] == 0.0


def test_drain_differs_from_close_at_campaign_scope():
    close = stopped(Scope.CAMPAIGN, Disposition.CLOSE)
    drain = stopped(Scope.CAMPAIGN, Disposition.DRAIN)
    assert bool(drain.draining) and not bool(close.draining)
