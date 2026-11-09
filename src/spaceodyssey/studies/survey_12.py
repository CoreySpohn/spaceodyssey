"""Survey 12: a failure sets an asset's ops state; a released record reports it.

Asset 0 (lanes 0 and 1) fails at `FAIL`: a process sets its ops state to 1 and writes
a failure notice that reaches the policy `delay` ticks later. An admission is debited
from the cost row of the state it finds, and a version-2 channel model realizes it
with that state's noise floor. A level-1 array policy starts the subject at each tick
of `starts` and wakes at each tick of `polls`. Three switches, all leaves, select what
it does:

- `react`: once the notice is in its snapshot, re-point the next observation to
  subject 1;
- `read_balance`: do the same when a debit differs from the first debit it saw (needs
  `Capacities.snapshot_balances`, which every study here has);
- `cancel_tick`: cancel lane 1 at that tick (a refund of the admission it holds).

Studies that differ only in these numbers share one compiled program.
"""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.declarations import Declaration
from spaceodyssey.core.enums import F64, I64, Audience, EffectKind, RecordStatus
from spaceodyssey.core.kinds import AbstractChannelModel, AbstractPolicy
from spaceodyssey.core.ledger import set_ops_state
from spaceodyssey.core.lift import traced
from spaceodyssey.core.snapshot import effects_none
from spaceodyssey.studies import toy
from spaceodyssey.studies.survey_11 import ScriptedAlerts

__all__ = [
    "BALANCE",
    "COSTS",
    "DELAY",
    "FAIL",
    "FAILURE_CHANNEL",
    "FLOORS",
    "DegradedFloorPhotometry",
    "FailureWatch",
    "OpsFailure",
    "survey_12",
]

FAIL = 100  # the tick asset 0 moves to ops state 1
DELAY = 25  # the failure notice reaches the policy this long after it is written
COSTS = (2.0, 5.0)  # the cost of every channel in ops state 0 and in ops state 1
FLOORS = (0.1, 0.5)  # the noise floor of a realization in ops state 0 and in 1
BALANCE = 100.0
FAILURE_CHANNEL = 50
N_EFFECTS = 3  # observe, wait, and the slot the policy's cancel takes
_FAR = 2**40


class OpsFailure(ScriptedAlerts):
    """A scripted record, released `delay` ticks late, that moves an asset's ops state.

    At each fire the process writes its record (the failure notice) and sets the ops
    state of `asset` to `state` in the same handler.
    """

    asset: int = eqx.field(default=0, metadata=traced(I64))
    state: int = eqx.field(default=1, metadata=traced(I64))

    def handle(self, carry, ev, study):
        """Write the notice and set the asset's ops state."""
        carry, rec = super().handle(carry, ev, study)
        carry = set_ops_state(carry, self.asset, self.state, study.caps)
        return carry, rec


class DegradedFloorPhotometry(AbstractChannelModel):
    """The toy signal with a noise floor chosen by the record's admission ops state.

    Version 2: the row is `[value, variate, fraction]` with
    `value = gain * signal + floors[ops_state] * variate`.
    """

    floors: jax.Array = eqx.field(converter=lambda x: jnp.asarray(x, F64))
    version: int = eqx.field(default=2, kw_only=True)

    @classmethod
    def migrate(cls, from_version, params):
        """A version-1 document carries no field this model needs to change."""
        return params

    def realize(
        self, truth, subject, channel_index, fraction, key, design, channels, *, context
    ):
        """Return the row; the floor is read from the stored admission state."""
        z = jr.normal(key, dtype=F64)
        value = design.gain * truth.signal[subject] + self.floors[context.ops_state] * z
        return jnp.stack([value, z, jnp.asarray(context.fraction, F64)])


class FailureWatch(AbstractPolicy):
    """Start the subject at `starts`, wake at `polls`, and re-point once a change shows.

    Row 0 of an effect table observes: at a start tick, the channel of that start
    (`channels`) on subject 0, or, once a change was noticed, on subject 1 (and at the
    tick the change is noticed, channel index 0). Row 1 waits for the next start or poll
    tick, and row 2 cancels lane 1 at `cancel_tick`. The change is noticed through
    the failure notice in the snapshot (`react`) or through a debit that differs from
    the first debit seen (`read_balance`); with both off the plan is fixed.

    Policy state: 0 noticed, 1 the last balance of asset 0, 2 the first debit seen.
    """

    starts: tuple[int, ...] = eqx.field(metadata=traced(I64, sequence=True))
    channels: tuple[int, ...] = eqx.field(metadata=traced(I64, sequence=True))
    polls: tuple[int, ...] = eqx.field(metadata=traced(I64, sequence=True))
    react: int = eqx.field(default=1, metadata=traced(I64))
    read_balance: int = eqx.field(default=0, metadata=traced(I64))
    cancel_tick: int = eqx.field(default=-1, metadata=traced(I64))

    def declare(self):
        """The program the observation is charged to."""
        return Declaration(programs=(0,))

    def decide(self, pstate, view, study, key):
        """Emit this tick's observation, wait and cancel."""
        caps = study.caps
        t = view.tick
        starts = jnp.asarray(self.starts, I64)
        wakes = jnp.concatenate([starts, jnp.asarray(self.polls, I64)])
        begin = (starts == t).any()
        until = jnp.where(wakes > t, wakes, _FAR).min()
        notice = ((view.log.channel == FAILURE_CHANNEL) & view.log.valid).any()
        balance = view.balance[0, 0]
        debit = jnp.where(view.n_dec > 0, pstate[1] - balance, 0.0)
        first = pstate[2]
        odd = (debit > 0) & (first > 0) & (jnp.abs(debit - first) > 1e-9)
        noticed = pstate[0] > 0
        now = ((self.react > 0) & notice) | ((self.read_balance > 0) & odd)
        newly = now & ~noticed
        seen = noticed | now
        subject = jnp.where(seen, 1, 0)
        chan = jnp.asarray(self.channels, I64)[jnp.argmax(starts == t)]
        channel = jnp.where(begin, chan, 0)
        cancel = (self.cancel_tick >= 0) & (t == self.cancel_tick)
        eff = effects_none(caps)
        kind = jnp.asarray(
            [
                jnp.where(begin | newly, EffectKind.OBSERVE, EffectKind.NONE),
                jnp.where((wakes > t).any(), EffectKind.WAIT, EffectKind.NONE),
                jnp.where(cancel, EffectKind.CANCEL, EffectKind.NONE),
            ],
            I64,
        )
        lanes = study.channels.lane[jnp.asarray([channel, 0, 0])].at[2].set(1)
        eff = dataclasses.replace(
            eff,
            kind=kind,
            subject=jnp.asarray([subject, 0, 0], I64),
            channel_index=jnp.asarray([channel, 0, 0], I64),
            group=t * N_EFFECTS + jnp.arange(N_EFFECTS, dtype=I64),
            tick=jnp.asarray([0, until, 0], I64),
            lane=lanes,
        )
        state = pstate.at[0].set(jnp.where(seen, 1.0, 0.0))
        state = state.at[1].set(balance)
        state = state.at[2].set(jnp.where((first <= 0) & (debit > 0), debit, first))
        return eff, state, jnp.zeros((), F64)


def survey_12(
    *,
    costs=COSTS,
    floors=FLOORS,
    fail=FAIL,
    delay=DELAY,
    starts=(0, 105, 120),
    channels=(0, 1, 0),
    polls=(110, 150),
    react=1,
    read_balance=0,
    cancel_tick=-1,
):
    """The toy study with ops states, per-state costs and floors, a failing asset 0.

    Every channel costs `costs[s]` per resource in state `s`; budgets are large, so no
    admission is refused. Refunds are on, and the policy sees balances.
    """
    caps = Capacities(
        n_effects=N_EFFECTS, refunds=True, n_ops=2, snapshot_balances=True
    )
    base = toy.survey_1(caps=caps)
    shape = (caps.n_channels, caps.n_resources)
    channels_table = dataclasses.replace(
        base.channels,
        cost=jnp.full(shape, costs[0], F64),
        charge=jnp.full(shape, costs[0], F64),
        cost_by_ops=jnp.stack([jnp.full(shape, c, F64) for c in costs]),
    )
    process = OpsFailure(
        ticks=(fail,),
        values=(1.0,),
        ids=(0,),
        revisions=(0,),
        statuses=(int(RecordStatus.FAILED),),
        kinds=(toy.RECORD_AUXILIARY,),
        audience=int(Audience.POLICY),
        delay=delay,
    )
    policy = FailureWatch(
        starts=starts,
        channels=channels,
        polls=polls,
        react=react,
        read_balance=read_balance,
        cancel_tick=cancel_tick,
    )
    study = toy.survey_1(
        caps=caps,
        channels=channels_table,
        budget=jnp.full((caps.n_balance_rows, caps.n_resources), BALANCE, F64),
        policies=(policy,),
        process=process,
        channel_model=DegradedFloorPhotometry(floors=floors),
    )
    return eqx.tree_at(lambda s: s.agenda.levels, study, jnp.asarray([1]))
