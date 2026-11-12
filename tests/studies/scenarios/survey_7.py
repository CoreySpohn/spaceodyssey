"""Bundled observations: two channels share one lane, one pointing and one background.

A bundle runs several observations of one lane as one occupancy. This survey puts the
toy's two channels on lane 2 (channel 12 holds the lane 50 ticks, channel 13 holds it
40) into one bundle, and shows three consequences.

- The group is admitted or refused whole: with the balance short of both members, a
  refusal of the pair is a refusal of each member.
- Pointing is charged once: the bundle starts after one transition and keeps the lane
  busy until its longest member ends. `PairPolicy` with `bundled` false requests the
  same two channels one after the other, so the lane changes pointing twice and the
  window ends later; it is the serializing control.
- The shared background is counted once. `BundleBackgroundObserver` reads each
  measurement as the signal plus noise of variance `noise_var` plus a background of
  variance `background_var` common to the members of one bundle. The members of a
  bundle (grouped by `Record.bundle`) together carry the information of
  `1 / (noise_var / m + background_var)`, with `m` the members folded so far; an
  observer that ignored the bundle would count the background once per member.
"""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp

from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.declarations import Declaration
from spaceodyssey.core.enums import (
    F64,
    I64,
    Disposition,
    EffectKind,
    RecordStatus,
    Scope,
)
from spaceodyssey.core.kinds import AbstractObserver, AbstractPolicy
from spaceodyssey.core.snapshot import BeliefLayout
from spaceodyssey.studies import builder, effect_rows, toy

LANE = 2
MEMBERS = (12, 13)  # channel ids, on lane 2
NOISE_VAR, BACKGROUND_VAR = 0.25, 0.5


def survey_7_caps():
    """Four channels (two on lane 2) and one bundle of two members per decision."""
    return Capacities(n_channels=4, max_bundles=1, max_bundle_size=2)


def _as_i64(x):
    return jnp.asarray(x, I64)


class BackgroundBelief(eqx.Module):
    """Per subject: information, measurements folded, and the open bundle.

    `prec` is the information about the subject's signal; `bundle` and `members` name
    the bundle the latest measurement joined and how many of its members were folded.
    """

    prec: jax.Array
    n: jax.Array
    bundle: jax.Array
    members: jax.Array
    aux: jax.Array


def project_background(belief):
    """The trace projection of a `BackgroundBelief`: its information and `aux`."""
    return belief.prec, belief.aux


class BundleBackgroundObserver(AbstractObserver):
    """Information about each subject, with the background shared inside a bundle.

    A fold of a member of the open bundle replaces the bundle's term
    `1 / (noise_var / m + background_var)` by the term for `m + 1` members; a record
    that opens a bundle, or has none, adds the term for one member. With `grouped`
    false every record opens its own term, the two-background belief. The members of
    one bundle are folded one after another, as the toy's two members are.
    """

    noise_var: float = eqx.field(default=NOISE_VAR)
    background_var: float = eqx.field(default=BACKGROUND_VAR)
    grouped: bool = eqx.field(default=True, static=True)

    def init(self, caps):
        """No information, no members, no open bundle."""
        n = caps.n_subjects
        return BackgroundBelief(
            prec=jnp.zeros(n, F64),
            n=jnp.zeros(n, I64),
            bundle=jnp.full(n, -1, I64),
            members=jnp.zeros(n, I64),
            aux=jnp.zeros(caps.n_an, F64),
        )

    def layout(self, caps):
        """The belief is its own summary, counted by `n`."""
        shape = jax.eval_shape(lambda: self.init(caps))
        return BeliefLayout(
            belief=shape, summary=shape, count_leaf="n", project=project_background
        )

    def _term(self, members):
        return 1.0 / (self.noise_var / members.astype(F64) + self.background_var)

    def fold(self, belief, rec, caps, key=None):
        """Add the measurement's information, once per bundle's background."""
        t = rec.subject
        take = (
            rec.valid
            & (rec.kind == toy.RECORD_MEASUREMENT)
            & (rec.status == RecordStatus.COMPLETED)
        )
        joins = (rec.bundle >= 0) & (rec.bundle == belief.bundle[t])
        if not self.grouped:
            joins = jnp.zeros_like(joins)
        members = jnp.where(joins, belief.members[t] + 1, 1)
        before = jnp.where(joins, self._term(belief.members[t]), 0.0)
        gain = self._term(members) - before
        return BackgroundBelief(
            prec=belief.prec.at[t].add(jnp.where(take, gain, 0.0)),
            n=belief.n.at[t].add(take.astype(I64)),
            bundle=belief.bundle.at[t].set(
                jnp.where(take, rec.bundle, belief.bundle[t])
            ),
            members=belief.members.at[t].set(
                jnp.where(take, members, belief.members[t])
            ),
            aux=belief.aux,
        )


class PairPolicy(AbstractPolicy):
    """Observe both members of the pair on lane 2 once, then drain.

    With `bundled` the first decision requests both members as one bundle (one
    admission group, ordinal 0) and the next decision drains. Without it the first
    decision requests the first member alone and the decision that sees its record
    requests the second, so the lane points twice. `subject` is the target.
    """

    bundled: jax.Array = eqx.field(default=1, converter=_as_i64)
    subject: int = eqx.field(default=0, static=True)
    members: tuple = eqx.field(default=MEMBERS, static=True)

    def declare(self):
        """Observe and the closing drain; the bundle's members when bundled."""
        bundles = ()
        if int(self.bundled):
            bundles = tuple((0, LANE, 0, 0, 0) for _ in self.members)
        return Declaration(
            effects=frozenset((EffectKind.OBSERVE, EffectKind.STOP)),
            stops=((Scope.CAMPAIGN, Disposition.DRAIN),),
            bundles=bundles,
        )

    def decide(self, pstate, view, study, key):
        """Request the pair at once or one member at a time, then drain."""
        bundled = self.bundled > 0
        first = view.n_dec == 0
        seen = view.belief.n[self.subject]
        pair = bundled & first
        one = ~bundled & first
        second = ~bundled & ~first & (seen == 1)
        stop = jnp.where(bundled, ~first, seen >= 2)
        act = pair | one | second
        kind0 = jnp.where(
            act, EffectKind.OBSERVE, jnp.where(stop, EffectKind.STOP, EffectKind.NONE)
        )
        kind1 = jnp.where(pair, EffectKind.OBSERVE, EffectKind.NONE)
        channel = jnp.stack(
            [
                effect_rows.chan_index(study, self.members[0]),
                effect_rows.chan_index(study, self.members[1]),
            ]
        )
        channel = channel.at[0].set(jnp.where(second, channel[1], channel[0]))
        eff = effect_rows.effects(
            study.caps,
            kind=jnp.stack([kind0, kind1]).astype(I64),
            subject=self.subject,
            channel_index=channel,
            group=view.n_dec * study.caps.n_effects,
            scope=Scope.CAMPAIGN,
            disposition=Disposition.DRAIN,
        )
        ordinal = jnp.where(pair, 0, -1) * jnp.ones(study.caps.n_effects, I64)
        eff = dataclasses.replace(eff, bundle=ordinal.astype(I64))
        return eff, pstate, jnp.asarray(0.0, F64)


def survey_7(bundled=True, **overrides):
    """The toy study with the pair policy and the background observer."""
    overrides.setdefault("policies", (PairPolicy(bundled=int(bundled)),))
    overrides.setdefault("observer", BundleBackgroundObserver())
    return builder.survey_1(survey_7_caps(), **overrides)


def short_balance(study, asset=1):
    """`study` with `asset`'s time resource holding one member's cost, not two."""
    return dataclasses.replace(study, budget=study.budget.at[asset, 0].set(1.0))


def pointing_end(study, start):
    """The tick a bundle started at `start` on lane 2 frees the lane, by the rule.

    One transition, then the longest member's occupancy.
    """
    rows = [list(map(int, study.channels.id)).index(c) for c in MEMBERS]
    longest = int(jnp.max(study.channels.occupancy_ticks[jnp.asarray(rows)]))
    return start + int(study.assets.transition_ticks[LANE]) + longest


def one_background(study, members=2):
    """The information of `members` bundled measurements with one background term."""
    obs = study.observer
    return 1.0 / (obs.noise_var / members + obs.background_var)


def two_backgrounds(study, members=2):
    """The information of `members` measurements counting the background each."""
    obs = study.observer
    return members / (obs.noise_var + obs.background_var)


__all__ = [
    "BACKGROUND_VAR",
    "LANE",
    "MEMBERS",
    "NOISE_VAR",
    "BackgroundBelief",
    "BundleBackgroundObserver",
    "PairPolicy",
    "one_background",
    "pointing_end",
    "short_balance",
    "survey_7",
    "survey_7_caps",
    "two_backgrounds",
]
