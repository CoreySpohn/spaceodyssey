"""Survey 11: an alert reaches the policy mid-exposure and the policy cancels the work.

Asset 0, lane 0 runs channel 10: 100 ticks of occupancy after 10 of transition, a
release delay of 20, and a price that differs from its cost in every resource column. A
scripted alert is written at `ALERT_FIRE` and released to the policy `ALERT_DELAY` ticks
later. A level-1 array policy scans the released rows of its snapshot for a row on the
alert channel and, once, emits a live `Cancel` of lane 0. It starts the subject at each
tick of `STARTS`, so the second start shows what became of the cancelled index.
"""

import dataclasses

import equinox as eqx
import jax.numpy as jnp

from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.declarations import Declaration
from spaceodyssey.core.enums import (
    F64,
    I64,
    Audience,
    ChargePricing,
    EffectKind,
    RecordStatus,
)
from spaceodyssey.core.kinds import AbstractPolicy
from spaceodyssey.core.lift import traced
from spaceodyssey.core.snapshot import effects_none
from spaceodyssey.studies import toy
from spaceodyssey.studies.scripted import SCRIPTED_CHANNEL, ScriptedRecords

ALERT_CHANNEL = SCRIPTED_CHANNEL
LEVEL = 1  # the policy level of the study: it reads arrays and may Cancel
N_EFFECTS = 3  # the observe, the wait, and the slot the policy's Cancel takes
LANE, SUBJECT, CHANNEL_INDEX = 0, 0, 0  # channel id 10 is index 0, on lane 0

ALERT_FIRE, ALERT_DELAY = 40, 15
STARTS = (0, 120)
HORIZON = 30  # the freeze horizon of asset 0 in the frozen variant
# Price of channel 10 per resource column: asset, program, shared. The cost is 10 in
# every column, so a cancel that credits the cost and not the charge differs from one
# that follows the table.
COST = 10.0
CHARGE = (20.0, 12.0, 8.0)
SURCHARGE = (2.0, 0.0, 0.0)
PRICING = (
    ChargePricing.SURCHARGE,
    ChargePricing.OCCUPANCY,
    ChargePricing.FLAT_FRACTION,
)
BALANCE = 100.0
_FAR = 2**40


class ScriptedAlerts(ScriptedRecords):
    """Scripted records released `delay` ticks after they are written.

    `ScriptedRecords` releases a record in the tick it is written. An alert is declared
    with a release delay: written at its fire tick, released to the policy `delay` ticks
    later. The delay is a leaf, so studies that differ only in it share a program.
    """

    delay: int = eqx.field(default=0, metadata=traced(I64))

    def handle(self, carry, ev, study):
        """Write the scripted record, available `delay` ticks after the fire."""
        carry, rec = super().handle(carry, ev, study)
        return carry, dataclasses.replace(rec, available_tick=ev.tick + self.delay)


class AlertWatch(AbstractPolicy):
    """Start the subject at each tick of `starts` and cancel the lane on an alert.

    A level-1 array policy. Effect row 0 observes the subject at a start tick, row 1
    waits for the next start tick, and row 2 is a live `Cancel(lane)` put there once,
    when a row on the alert channel is in the snapshot. With `disruptive` the starts
    break the frozen horizon (the study then holds commitments).
    """

    starts: tuple[int, ...] = eqx.field(metadata=traced(I64, sequence=True))
    disruptive: bool = eqx.field(default=False, static=True)

    def declare(self):
        """The program the observe is charged to, and a disruptive start where asked."""
        return Declaration(programs=(0,), disruptive=self.disruptive)

    def decide(self, pstate, view, study, key):
        """Emit the start and wait of this tick, then the Cancel once an alert shows."""
        caps = study.caps
        t = view.tick
        starts = jnp.asarray(self.starts, I64)
        begin = (starts == t).any()
        later = (starts > t).any()
        until = jnp.where(starts > t, starts, _FAR).min()
        alert = ((view.log.channel == ALERT_CHANNEL) & view.log.valid).any()
        due = alert & (pstate[0] == 0.0)
        eff = effects_none(caps)
        kind = jnp.asarray(
            [
                jnp.where(begin, EffectKind.OBSERVE, EffectKind.NONE),
                jnp.where(later, EffectKind.WAIT, EffectKind.NONE),
                jnp.where(due, EffectKind.CANCEL, EffectKind.NONE),
            ],
            I64,
        )
        extra = {}
        if caps.commitments:
            extra = {
                "start_tick": jnp.full(N_EFFECTS, t, I64),
                "disruptive": jnp.asarray([self.disruptive, False, False]),
            }
        eff = dataclasses.replace(
            eff,
            kind=kind,
            subject=jnp.full(N_EFFECTS, SUBJECT, I64),
            channel_index=jnp.full(N_EFFECTS, CHANNEL_INDEX, I64),
            group=t * N_EFFECTS + jnp.arange(N_EFFECTS, dtype=I64),
            tick=jnp.asarray([0, until, 0], I64),
            lane=jnp.full(N_EFFECTS, LANE, I64),
            **extra,
        )
        return eff, pstate.at[0].set(jnp.where(due, 1.0, pstate[0])), jnp.zeros((), F64)


def survey_study(*, frozen=False):
    """The toy study with refunds on, the priced channel, the alert and the policy.

    Every balance row holds `BALANCE`, so a debit never refuses. With `frozen` the
    capacities hold commitments, asset 0 has a freeze horizon and names resource column
    0 as its disruptive resource, and the policy's starts break the freeze, so the live
    cancel meets a frozen asset. Awards stay off. The agenda is at level 1.
    """
    caps = Capacities(
        n_effects=N_EFFECTS, refunds=True, **({"n_commitments": 4} if frozen else {})
    )
    alerts = ScriptedAlerts(
        ticks=(ALERT_FIRE,),
        values=(1.0,),
        ids=(0,),
        revisions=(0,),
        statuses=(int(RecordStatus.COMPLETED),),
        audience=int(Audience.POLICY),
        delay=ALERT_DELAY,
    )
    base = toy.survey_1(caps=caps)
    c, r = caps.n_channels, caps.n_resources
    cost = jnp.full((c, r), COST, F64)
    table = jnp.full((c, r), ChargePricing.OCCUPANCY, I64)
    channels = dataclasses.replace(
        base.channels,
        cost=cost,
        charge=cost.at[CHANNEL_INDEX].set(jnp.asarray(CHARGE, F64)),
        pricing=table.at[CHANNEL_INDEX].set(jnp.asarray(PRICING, I64)),
        surcharge=jnp.zeros((c, r), F64)
        .at[CHANNEL_INDEX]
        .set(jnp.asarray(SURCHARGE, F64)),
    )
    study = toy.survey_1(
        caps=caps,
        channels=channels,
        budget=jnp.full((caps.n_balance_rows, r), BALANCE, F64),
        policies=(AlertWatch(starts=STARTS, disruptive=frozen),),
        process=alerts,
    )
    if frozen:
        assets = dataclasses.replace(
            study.assets,
            freeze_ticks=jnp.asarray([HORIZON, 0], I64),
            disruptive_resource=jnp.asarray([0, -1], I64),
        )
        study = dataclasses.replace(study, assets=assets)
    return eqx.tree_at(lambda s: s.agenda.levels, study, jnp.asarray([LEVEL]))
