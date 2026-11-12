"""A phased mission: one policy up to a prescribed tick, another after it.

The switch is the integer phase boundary `Agenda.phase_until`; phase selection compares
integer ticks. A boundary of `caps.max_tick` never switches: it is the canonical
"no switch" value a document writes as `{"no_switch": true}`, and the study then runs
as `survey_1` does.
"""

import jax.numpy as jnp

from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.enums import I64
from spaceodyssey.studies import toy

# The first phase repeats `survey_1`'s plan; the second revisits both subjects on the
# second channel, so a switch changes the acquisitions that follow it.
SURVEY_PLAN = ((0, 10), (1, 10), (0, 10), (1, 10), (0, 10))
SECOND_PHASE_PLAN = ((1, 11), (0, 11), (1, 11))


def no_switch(caps=None):
    """The phase boundary that never switches: the capacity's last tick."""
    caps = toy.toy_registry().lower(Capacities()) if caps is None else caps
    return caps.max_tick


def survey_14(switch_tick=None, caps=None, **overrides):
    """Build `survey_1` with a second phase that starts at `switch_tick`.

    `switch_tick=None` is the infinite threshold: both boundaries hold
    `caps.max_tick`, so the second phase never runs. A finite `switch_tick` is the
    tick from which the second phase's policy decides (a decision at that tick is its
    own). Other keyword arguments go to `toy.survey_1`.
    """
    caps = toy.toy_registry().lower(Capacities()) if caps is None else caps
    if not caps.payload_audience:
        caps = toy.toy_registry().lower(caps)
    first = int(caps.max_tick if switch_tick is None else switch_tick)
    policies = (
        toy.FixedPlanPolicy(plan=SURVEY_PLAN),
        toy.FixedPlanPolicy(plan=SECOND_PHASE_PLAN),
    )
    overrides.setdefault("policies", policies)
    overrides.setdefault("phase_until", jnp.asarray((first, caps.max_tick), I64))
    return toy.survey_1(caps=caps, **overrides)


__all__ = ["SECOND_PHASE_PLAN", "SURVEY_PLAN", "no_switch", "survey_14"]
