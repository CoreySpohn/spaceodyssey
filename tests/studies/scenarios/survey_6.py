"""Aging ephemerides: a value that reads the age of the accepted prior record.

A pre-mission table brings a public prior of subject 0 (the planet's ephemeris) and
a raw row of subject 1 into the run at tick 0, both observed before the run starts.
The prior's age at a decision is the decision tick less the prior row's `end_tick`;
`AgingValuePolicy` values the ephemeris as `base * decay ** age` and waits, from its
first decision, until that value would reach `threshold` (at least one tick, at most
`max_wait`), then observes the planet and drains the campaign. Every tick the policy
reads is relative (a released row's time against the decision tick), so shifting the
epoch origin changes nothing; doubling the prior's age moves the value monotonically.

The policy keeps the value of its first decision in `pstate[0]`. With `absolute` set
(a control) it takes the age as minus the prior's end tick, as if the run started at
tick 0: both readings agree on an unshifted run and differ on a shifted one.
"""

import equinox as eqx
import jax
import jax.numpy as jnp

from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.declarations import Declaration
from spaceodyssey.core.enums import (
    F64,
    I64,
    PUBLIC,
    Disposition,
    EffectKind,
    Scope,
)
from spaceodyssey.core.kinds import AbstractPolicy
from spaceodyssey.core.records import KindRegistry, RecordKind
from spaceodyssey.studies import toy
from spaceodyssey.studies.premission import PreMissionImporter, authored_table
from spaceodyssey.studies.prior_observer import PRIOR_KIND, PriorObserver

PRIOR_INDEX = 3  # the prior kind's index in `survey_6_registry`
RAW_CHANNEL, PRIOR_CHANNEL = 60, 61  # the channels of the imported rows
PLANET = 0  # the subject whose prior is the ephemeris
AGE = 10  # the prior's age at tick 0: its end offset is -AGE, its start -2 AGE
OBSERVE_CHANNEL = 10
N_OBSERVATIONS = 2
LOW_TICK = -(2**62)  # below any tick


def _f64(x):
    return jnp.asarray(x, F64)


def survey_6_registry():
    """The toy kinds and the prior kind, at indices 0 to 3."""
    return KindRegistry(
        (RecordKind("none", ()), toy.MEASUREMENT_KIND, toy.AUXILIARY_KIND, PRIOR_KIND)
    )


class AgingValuePolicy(AbstractPolicy):
    """Value the planet's ephemeris by its age; observe when the value runs down.

    The first decision reads the released prior rows about `subject` and takes the
    newest `end_tick`; the age is the decision tick less it (minus it with `absolute`
    set, a control) and the value `base * decay ** age` (0 with no prior). It waits
    the ticks the value would take to reach `threshold` at that rate, at least one
    and at most `max_wait`, and keeps the value in `pstate[0]`. The next `n_obs`
    decisions observe `subject`; then the campaign drains. It draws nothing.
    """

    base: jax.Array = eqx.field(default=1.0, converter=_f64)
    decay: jax.Array = eqx.field(default=0.9, converter=_f64)
    threshold: jax.Array = eqx.field(default=0.2, converter=_f64)
    absolute: jax.Array = eqx.field(
        default=False, converter=lambda x: jnp.asarray(x, bool)
    )
    subject: int = eqx.field(default=PLANET, static=True)
    prior_kind: int = eqx.field(default=PRIOR_INDEX, static=True)
    channel: int = eqx.field(default=OBSERVE_CHANNEL, static=True)
    n_obs: int = eqx.field(default=N_OBSERVATIONS, static=True)
    max_wait: int = eqx.field(default=400, static=True)

    def declare(self):
        """Observe, wait and the closing campaign drain."""
        return Declaration(
            effects=frozenset((EffectKind.OBSERVE, EffectKind.WAIT, EffectKind.STOP)),
            stops=((Scope.CAMPAIGN, Disposition.DRAIN),),
        )

    def value(self, view):
        """`(value, age, found)` of the newest released prior of `subject`."""
        log = view.log
        prior = (
            log.valid & (log.kind == self.prior_kind) & (log.subject == self.subject)
        )
        ended = jnp.max(jnp.where(prior, log.end_tick, LOW_TICK))
        now = jnp.where(self.absolute, jnp.asarray(0, I64), view.tick)
        age = now - ended
        found = prior.any()
        value = self.base * self.decay ** jnp.where(found, age, 0).astype(F64)
        return jnp.where(found, value, 0.0), age, found

    def wait(self, value):
        """Ticks until the value would reach the threshold, within `[1, max_wait]`."""
        safe = jnp.where(value > 0, value, self.threshold)
        ticks = jnp.ceil(jnp.log(self.threshold / safe) / jnp.log(self.decay))
        ticks = jnp.where(jnp.isfinite(ticks), ticks, 1.0)
        return jnp.clip(ticks, 1, self.max_wait).astype(I64)

    def decide(self, pstate, view, study, key):
        """Wait while the ephemeris holds its value, then observe, then drain."""
        first = view.n_dec == 0
        stop = view.n_dec > self.n_obs
        value, _, _ = self.value(view)
        act = jnp.where(stop, EffectKind.STOP, EffectKind.OBSERVE)
        channel = jnp.argmax(study.channels.id == self.channel).astype(I64)
        eff = toy._effects(
            study.caps,
            kind=jnp.stack(
                [
                    jnp.where(first, EffectKind.NONE, act),
                    jnp.where(first, EffectKind.WAIT, EffectKind.NONE),
                ]
            ).astype(I64),
            subject=self.subject,
            channel_index=channel,
            group=view.n_dec,
            tick=jnp.stack([jnp.asarray(0, I64), view.tick + self.wait(value)]),
            scope=Scope.CAMPAIGN,
            disposition=Disposition.DRAIN,
        )
        pstate = pstate.at[0].set(jnp.where(first, value, pstate[0]))
        return eff, pstate, jnp.asarray(0.0, F64)


def survey_6_table(age=AGE):
    """The prior of the planet, `age` ticks old at tick 0, and a raw row of subject 1.

    The prior spans `[-2 age, -age]` and the raw row `[-5, -1]`; both are public.
    """
    return authored_table(
        subject=[PLANET, 1],
        local_index=[0, 0],
        channel=[PRIOR_CHANNEL, RAW_CHANNEL],
        kind=[PRIOR_INDEX, toy.RECORD_MEASUREMENT],
        audience=[PUBLIC, PUBLIC],
        payload=[[1.0, 1.0], [2.0, 0.0]],
        start_offset=[-2 * age, -5],
        end_offset=[-age, -1],
        pedigree=[((7, 90, PLANET, 0, True),), ()],
    )


def survey_6(table=None, *, policy=None, **overrides):
    """The survey: the toy channels, the import, the prior observer, the value policy.

    `table` defaults to `survey_6_table()` and `policy` to `AgingValuePolicy()`; other
    overrides follow `toy.survey_1`.
    """
    table = survey_6_table() if table is None else table
    observer = PriorObserver(table=table, prior_kind=PRIOR_INDEX)
    caps = overrides.pop("caps", None) or Capacities()
    caps = survey_6_registry().lower(caps, observer)
    policy = AgingValuePolicy() if policy is None else policy
    return toy.survey_1(
        caps,
        process=PreMissionImporter(table=table),
        observer=observer,
        policies=(policy,),
        **overrides,
    )


__all__ = [
    "AGE",
    "PLANET",
    "PRIOR_INDEX",
    "AgingValuePolicy",
    "survey_6",
    "survey_6_registry",
    "survey_6_table",
]
