"""Exozodi-first (prescribed): a star-level precursor chooses between two plans.

The world has a group latent, one exozodi level per star, shared by the star's
planets: `Survey18Truth` holds the planets' signals and the stars' levels, drawn
from the world key (`Purpose.WORLD`) by `Survey18WorldSource`. Every measurement of a
planet depends on its star's level (`Survey18Photometry`).

A pre-mission table brings a precursor into the run: one raw row of the kind
`star_level` (`STAR_LEVEL_KIND`), a measurement of the target star's level made
before the mission, filed under the star's first planet. `Survey18Importer` writes
the row at tick 0 like the pre-mission importer and lets it be released later: a
row's `release_delay` puts its release that many ticks after the import, so a
precursor can be released at tick 0, at a later tick `T`, or after the last decision
(withheld).

`Survey18Observer` keeps the planets' running means and, in the auxiliary slots, the
posterior mean of each star's level: a Gaussian with a declared prior (mean
`prior_mean`, precision `prior_weight` in units of one precursor row) updated by the
star-level rows only. The policy summary is folded from the rows released to the
policy, so a precursor the policy never receives cannot reach it.

`Survey18Policy` chooses, at every decision, between two plans by its summary's mean
of the target star's level: below `threshold` (a faint disk) it observes the target
star's planets, otherwise the other star's. It reads nothing else of the belief, so
the plan depends on the world only through the released precursor rows.
"""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.declarations import Declaration
from spaceodyssey.core.enums import (
    F64,
    I64,
    PUBLIC,
    Disposition,
    EffectKind,
    RecordStatus,
    Scope,
)
from spaceodyssey.core.kinds import (
    AbstractChannelModel,
    AbstractObserver,
    AbstractPolicy,
    AbstractWorldSource,
)
from spaceodyssey.core.records import KindRegistry, PayloadColumn, RecordKind
from spaceodyssey.studies import builder, effect_rows, toy
from spaceodyssey.studies.premission import PreMissionImporter, authored_table

STAR_LEVEL_KIND = RecordKind("star_level", (PayloadColumn("value", PUBLIC),))
STAR_LEVEL_INDEX = 3  # its index in `survey_18_registry`

STAR_OF = (0, 0, 1, 1)  # the star of each planet (subject)
N_STARS = 2
TARGET_STAR = 0  # the star the precursor measures
PLANS = ((0, 1), (2, 3))  # the faint plan (target star's planets), the bright plan
PRECURSOR_CHANNEL = 70  # the channel of the imported precursor row
OBSERVE_CHANNEL = 10
N_OBSERVATIONS = 4
PRIOR_MEAN = 1.0  # the declared prior of a star's level
PRIOR_WEIGHT = 1.0  # its precision, in units of one precursor row
THRESHOLD = 0.8  # a summary mean below it chooses the faint plan
FAINT, BRIGHT = 0.2, 1.8  # precursor values on either side (posterior 0.6 and 1.4)
WITHHELD = 10**6  # a release delay past any decision of the survey


def _f64(x):
    return jnp.asarray(x, F64)


def _i64(x):
    return jnp.asarray(x, I64)


def survey_18_registry():
    """The toy kinds and the star-level kind, at indices 0 to 3."""
    return KindRegistry(
        (
            RecordKind("none", ()),
            toy.MEASUREMENT_KIND,
            toy.AUXILIARY_KIND,
            STAR_LEVEL_KIND,
        )
    )


class Survey18Truth(eqx.Module):
    """The planets' signals and each star's exozodi level (the group latent)."""

    signal: jax.Array
    level: jax.Array


def draw_survey_18_truth(key, caps, n_stars=N_STARS):
    """Draw a world from its key: signals in [0.5, 1.5], star levels in [0, 2]."""
    k_signal, k_level = jr.split(key)
    return Survey18Truth(
        signal=jr.uniform(k_signal, (caps.n_subjects,), minval=0.5, maxval=1.5),
        level=jr.uniform(k_level, (n_stars,), minval=0.0, maxval=2.0),
    )


class Survey18WorldSource(AbstractWorldSource):
    """Source of the planets' signals and the stars' shared levels."""

    n_stars: int = eqx.field(default=N_STARS, static=True)

    def draw(self, key, capacities):
        """Draw one world from the world key."""
        return draw_survey_18_truth(key, capacities, self.n_stars)


class Survey18Photometry(AbstractChannelModel):
    """A planet's signal plus its star's level, with Gaussian measurement noise.

    The value is `gain * signal[s] + coupling * level[star_of[s]] + noise * z`, so
    every planet of a star carries that star's level.
    """

    noise: jax.Array = eqx.field(converter=_f64)
    coupling: jax.Array = eqx.field(default=1.0, converter=_f64)
    star_of: tuple = eqx.field(default=STAR_OF, static=True)

    def realize(self, truth, subject, channel_index, fraction, key, design, channels):
        """Returns (measurement, raw standard normal variate)."""
        z = jr.normal(key, dtype=F64)
        noise = self.noise[channel_index] / jnp.sqrt(jnp.maximum(fraction, 1e-3))
        star = jnp.asarray(self.star_of, I64)[subject]
        signal = design.gain * truth.signal[subject] + self.coupling * truth.level[star]
        return signal + noise * z, z


class Survey18Belief(eqx.Module):
    """Planet means and counts, star levels in `aux`, star-level rows per star.

    `aux[k]` is the posterior mean of star `k`'s level for `k < n_stars` (0 beyond);
    `n_level[k]` counts the star-level rows folded for star `k`.
    """

    mean: jax.Array
    n: jax.Array
    aux: jax.Array
    n_level: jax.Array


class Survey18Observer(AbstractObserver):
    """Planet running means and a Gaussian posterior of each star's level.

    A measurement row updates its planet's running mean; a star-level row updates the
    level of the star of its subject, starting from the declared prior. Nothing else
    is read. The summary has the belief's type and is folded the same way, from the
    rows released to the policy.
    """

    prior_mean: jax.Array = eqx.field(default=PRIOR_MEAN, converter=_f64)
    prior_weight: jax.Array = eqx.field(default=PRIOR_WEIGHT, converter=_f64)
    star_of: tuple = eqx.field(default=STAR_OF, static=True)
    n_stars: int = eqx.field(default=N_STARS, static=True)
    measurement_kind: int = eqx.field(default=toy.RECORD_MEASUREMENT, static=True)
    level_kind: int = eqx.field(default=STAR_LEVEL_INDEX, static=True)

    def init(self, caps):
        """Zero planet means; every star's level at the declared prior mean."""
        if self.n_stars > caps.n_an:
            raise ValueError(
                f"{self.n_stars} star levels need as many auxiliary slots; the "
                f"capacities hold {caps.n_an}"
            )
        if len(self.star_of) != caps.n_subjects:
            raise ValueError(
                f"star_of names {len(self.star_of)} planets; the study holds "
                f"{caps.n_subjects}"
            )
        aux = jnp.zeros(caps.n_an, F64).at[: self.n_stars].set(self.prior_mean)
        return Survey18Belief(
            mean=jnp.zeros(caps.n_subjects, F64),
            n=jnp.zeros(caps.n_subjects, I64),
            aux=aux,
            n_level=jnp.zeros(self.n_stars, I64),
        )

    def fold(self, belief, rec, caps, key=None):
        """Fold a measurement into its planet, or a star-level row into its star."""
        s = rec.subject
        completed = rec.valid & (rec.status == RecordStatus.COMPLETED)
        take = completed & (rec.kind == self.measurement_kind)
        level = completed & (rec.kind == self.level_kind)
        n = belief.n[s].astype(F64)
        mean = (belief.mean[s] * n + rec.payload[0]) / (n + 1.0)
        star = jnp.asarray(self.star_of, I64)[s]
        held = self.prior_weight + belief.n_level[star].astype(F64)
        posterior = (belief.aux[star] * held + rec.payload[0]) / (held + 1.0)
        return Survey18Belief(
            mean=belief.mean.at[s].set(jnp.where(take, mean, belief.mean[s])),
            n=belief.n.at[s].add(take.astype(I64)),
            aux=belief.aux.at[star].set(jnp.where(level, posterior, belief.aux[star])),
            n_level=belief.n_level.at[star].add(level.astype(I64)),
        )


class Survey18Policy(AbstractPolicy):
    """Choose a plan by the summary's mean of the target star's level; follow it.

    At every decision the plan is `plans[0]` (the target star's planets) when
    `view.belief.aux[target]` is below `threshold`, else `plans[1]`. The decision
    observes the plan's planet at the position given by the measurements folded so
    far (`sum(view.belief.n)`), and drains the campaign once `n_obs` are folded. A
    decision while the lane is busy is refused and repeated at the next release. It
    draws nothing.
    """

    threshold: jax.Array = eqx.field(default=THRESHOLD, converter=_f64)
    target: int = eqx.field(default=TARGET_STAR, static=True)
    plans: tuple = eqx.field(default=PLANS, static=True)
    channel: int = eqx.field(default=OBSERVE_CHANNEL, static=True)
    n_obs: int = eqx.field(default=N_OBSERVATIONS, static=True)

    def declare(self):
        """Observe and the closing campaign drain."""
        return Declaration(
            effects=frozenset((EffectKind.OBSERVE, EffectKind.STOP)),
            stops=((Scope.CAMPAIGN, Disposition.DRAIN),),
        )

    def level(self, view):
        """The level the plan is chosen by: the summary's mean for the target."""
        return view.belief.aux[self.target]

    def decide(self, pstate, view, study, key):
        """Observe the chosen plan's next planet, or drain once the plan is done."""
        plans = jnp.asarray(self.plans, I64)
        choice = jnp.where(self.level(view) < self.threshold, 0, 1)
        done = jnp.sum(view.belief.n)
        subject = plans[choice, done % plans.shape[1]]
        stop = done >= self.n_obs
        channel = jnp.argmax(study.channels.id == self.channel).astype(I64)
        eff = effect_rows.effects(
            study.caps,
            kind=jnp.stack(
                [
                    jnp.where(stop, EffectKind.STOP, EffectKind.OBSERVE),
                    jnp.asarray(EffectKind.NONE, I64),
                ]
            ).astype(I64),
            subject=subject,
            channel_index=channel,
            group=view.n_dec,
            scope=Scope.CAMPAIGN,
            disposition=Disposition.DRAIN,
        )
        return eff, pstate, jnp.asarray(0.0, F64)


class Survey18Importer(PreMissionImporter):
    """The pre-mission importer with a release delay per row.

    Row `i` is written at tick 0, as the pre-mission importer writes it, and becomes
    available `release_delay[i]` ticks after its import, so its release event (and
    the stamps it sets) come that many ticks later. A delay of 0 is the pre-mission
    importer. The row's start and end are still relative to the import tick.
    """

    release_delay: jax.Array = eqx.field(converter=_i64)

    def __check_init__(self):
        """Require one nonnegative delay per row."""
        if jnp.shape(self.release_delay) != jnp.shape(self.table.local_index):
            raise ValueError("release_delay is one value per table row")
        if not isinstance(self.release_delay, jax.core.Tracer) and bool(
            np.any(np.asarray(self.release_delay) < 0)
        ):
            raise ValueError("a row is released at or after its import")

    def handle(self, carry, ev, study):
        """Write the row; it becomes available after its delay."""
        carry, rec = super().handle(carry, ev, study)
        if self.table.n_rows == 0:
            return carry, rec
        available = ev.tick + self.release_delay[ev.group]
        return carry, dataclasses.replace(rec, available_tick=available)


def survey_18_table(value=FAINT, audience=PUBLIC):
    """The precursor: one star-level row of the target star, measured before tick 0.

    It is filed under the target star's first planet; `value` is its payload and
    `audience` who receives it.
    """
    return authored_table(
        subject=[STAR_OF.index(TARGET_STAR)],
        local_index=[0],
        channel=[PRECURSOR_CHANNEL],
        kind=[STAR_LEVEL_INDEX],
        audience=[audience],
        payload=[[value]],
        start_offset=[-30],
        end_offset=[-10],
    )


def survey_18(
    table=None,
    *,
    release_delay=0,
    policy=None,
    observer=None,
    coupling=1.0,
    **overrides,
):
    """The survey: two stars of two planets, the precursor import, the level policy.

    `table` defaults to `survey_18_table()`; `release_delay` (one value, or one per
    row) delays its release. `policy` and `observer` replace `Survey18Policy()` and
    `Survey18Observer()` (controls); `coupling` is the weight of a star's level in
    its planets' measurements. Other overrides follow `builder.survey_1`.
    """
    table = survey_18_table() if table is None else table
    delay = jnp.broadcast_to(_i64(release_delay), jnp.shape(table.local_index))
    caps = overrides.pop("caps", None) or Capacities()
    caps = survey_18_registry().lower(caps)
    base = builder.survey_1(
        caps,
        process=Survey18Importer(table=table, release_delay=delay),
        observer=Survey18Observer() if observer is None else observer,
        policies=(Survey18Policy() if policy is None else policy,),
        **overrides,
    )
    noise = base.channel_model.noise
    return dataclasses.replace(
        base,
        channel_model=Survey18Photometry(noise=noise, coupling=coupling),
        world_source=Survey18WorldSource(),
    )


__all__ = [
    "BRIGHT",
    "FAINT",
    "N_OBSERVATIONS",
    "N_STARS",
    "PLANS",
    "PRECURSOR_CHANNEL",
    "PRIOR_MEAN",
    "PRIOR_WEIGHT",
    "STAR_LEVEL_INDEX",
    "STAR_LEVEL_KIND",
    "STAR_OF",
    "TARGET_STAR",
    "THRESHOLD",
    "WITHHELD",
    "Survey18Belief",
    "Survey18Importer",
    "Survey18Observer",
    "Survey18Photometry",
    "Survey18Policy",
    "Survey18Truth",
    "Survey18WorldSource",
    "draw_survey_18_truth",
    "survey_18",
    "survey_18_registry",
    "survey_18_table",
]
