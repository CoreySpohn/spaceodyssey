"""Astrometric candidates: a delayed catalog released to the observer and the policy.

The catalog channel (`catalog.CATALOG_CHANNEL_ID`) writes one catalog row per
acquisition, released `catalog.CATALOG_DELAY_TICKS` after its occupancy ends. Each row
puts one candidate on a roster the observer keeps. A candidate has an identifier of its
own, `CANDIDATE_ID_BASE` minus its place on the roster, which never equals a truth id
(a subject index). The policy reads the roster size from its summary; the candidate's
catalog value stays an observer-only column, so the policy never sees it.

The follow-up policy observes `follow_subject` once for each candidate it has not yet
followed, and otherwise continues its survey plan. A decision made before a candidate's
release tick therefore reads an empty roster, and the candidate cannot change it.
"""

import dataclasses

import equinox as eqx
import jax.numpy as jnp

from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.declarations import Declaration
from spaceodyssey.core.enums import F64, I64, PUBLIC, Disposition, EffectKind, Scope
from spaceodyssey.core.kinds import AbstractPolicy
from spaceodyssey.core.snapshot import MeanBelief
from spaceodyssey.studies import builder, catalog, effect_rows, toy

# aux slots of the belief: 0 and 1 are the toy's and the catalog's own
ROSTER_COUNT_SLOT = 2
LAST_CANDIDATE_SLOT = 3
# Negative ids are the candidate namespace; subjects and acquisitions are nonnegative.
CANDIDATE_ID_BASE = -1
# The survey plan: subject and channel id of each planned acquisition. The third entry
# is the catalog channel, whose row is the candidate.
SURVEY_PLAN = (
    (0, 10),
    (0, catalog.CATALOG_CHANNEL_ID),
    (1, 10),
    (0, 10),
    (1, 10),
)
FOLLOW_SUBJECT = 1
FOLLOW_CHANNEL_ID = 11


class RosterObserver(catalog.CatalogObserver):
    """The catalog observer, with the candidates on a roster.

    A catalog row adds one candidate: the roster count (`ROSTER_COUNT_SLOT`) rises by
    one and the identifier of the newest candidate (`LAST_CANDIDATE_SLOT`) is
    `CANDIDATE_ID_BASE` minus its place on the roster. Nothing reads the row's payload,
    so the roster is the same for an audience that cannot read the catalog value.
    """

    def fold(self, belief, rec, caps, key=None):
        """Fold as the catalog observer does, then add the row's candidate."""
        belief = super().fold(belief, rec, caps, key=key)
        row = rec.valid & (rec.kind == self.catalog_kind)
        count = belief.aux[ROSTER_COUNT_SLOT] + jnp.where(row, 1.0, 0.0)
        last = jnp.where(
            row,
            CANDIDATE_ID_BASE - belief.aux[ROSTER_COUNT_SLOT],
            belief.aux[LAST_CANDIDATE_SLOT],
        )
        aux = belief.aux.at[ROSTER_COUNT_SLOT].set(count)
        aux = aux.at[LAST_CANDIDATE_SLOT].set(last)
        return MeanBelief(belief.mean, belief.n, aux)


class FollowUpPolicy(AbstractPolicy):
    """Follow a survey plan; follow each candidate on the roster up once.

    The roster size is `view.belief.aux[ROSTER_COUNT_SLOT]`. When it exceeds the
    candidates already followed (`pstate[1]`) the decision observes `follow_subject` on
    `follow_channel` and leaves the plan where it was; otherwise it takes the next plan
    entry, and drains the campaign when the plan is spent. The decision index is the
    effect group. The policy draws nothing.
    """

    plan: tuple = eqx.field(default=SURVEY_PLAN, static=True)
    follow_subject: int = eqx.field(default=FOLLOW_SUBJECT, static=True)
    follow_channel: int = eqx.field(default=FOLLOW_CHANNEL_ID, static=True)

    def declare(self):
        """The closing campaign drain."""
        return Declaration(
            effects=frozenset((EffectKind.OBSERVE, EffectKind.STOP)),
            stops=((Scope.CAMPAIGN, Disposition.DRAIN),),
        )

    def decide(self, pstate, view, study, key):
        """Follow a new candidate up, else continue the survey plan."""
        caps = study.caps
        plan = jnp.asarray(self.plan, I64)
        pos = pstate[0].astype(I64)
        followed = pstate[1]
        due = view.belief.aux[ROSTER_COUNT_SLOT] > followed
        spent = pos >= plan.shape[0]
        at = jnp.minimum(pos, plan.shape[0] - 1)
        ids = study.channels.id
        survey_channel = jnp.argmax(ids == plan[at, 1]).astype(I64)
        follow_channel = jnp.argmax(ids == self.follow_channel).astype(I64)
        observe = due | ~spent
        kind0 = jnp.where(observe, EffectKind.OBSERVE, EffectKind.STOP)
        eff = effect_rows.effects(
            caps,
            kind=jnp.stack([kind0, jnp.asarray(EffectKind.NONE, I64)]),
            subject=jnp.where(due, self.follow_subject, plan[at, 0]),
            channel_index=jnp.where(due, follow_channel, survey_channel),
            group=view.n_dec,
            scope=Scope.CAMPAIGN,
            disposition=Disposition.DRAIN,
        )
        advance = ~due & ~spent
        new_ps = pstate.at[0].add(advance.astype(F64)).at[1].add(due.astype(F64))
        return eff, new_ps, jnp.asarray(0.0, F64)


class AbsolutePlanPolicy(AbstractPolicy):
    """Observe the plan at fixed absolute ticks, whatever has been released.

    Entry `i` of `plan` (subject, channel id) is observed at tick `starts[i]` and the
    policy then waits for the next start; the last entry also ends the campaign. A
    decision woken by a release at any other tick does nothing, so the schedule of
    acquisitions does not depend on when records are released. This is the plan the
    replay-with-latency check needs.
    """

    starts: tuple = eqx.field(default=(0, 150, 300, 450, 600), static=True)
    plan: tuple = eqx.field(default=SURVEY_PLAN, static=True)

    def __check_init__(self):
        """One strictly increasing start per plan entry."""
        if len(self.starts) != len(self.plan):
            raise ValueError("starts needs one tick per plan entry")
        if any(b <= a for a, b in zip(self.starts, self.starts[1:], strict=False)):
            raise ValueError("starts must increase")

    def declare(self):
        """The closing campaign drain."""
        return Declaration(
            effects=frozenset((EffectKind.OBSERVE, EffectKind.WAIT, EffectKind.STOP)),
            stops=((Scope.CAMPAIGN, Disposition.DRAIN),),
        )

    def decide(self, pstate, view, study, key):
        """Observe at a planned tick, and wait for the next one."""
        starts = jnp.asarray(self.starts, I64)
        plan = jnp.asarray(self.plan, I64)
        here = starts == view.tick
        at_start = jnp.any(here)
        i = jnp.argmax(here)
        last = i == len(self.starts) - 1
        following = starts[jnp.minimum(i + 1, len(self.starts) - 1)]
        channel = jnp.argmax(study.channels.id == plan[i, 1]).astype(I64)
        second = jnp.where(last, EffectKind.STOP, EffectKind.WAIT)
        eff = effect_rows.effects(
            study.caps,
            kind=jnp.stack(
                [
                    jnp.where(at_start, EffectKind.OBSERVE, EffectKind.NONE),
                    jnp.where(at_start, second, EffectKind.NONE).astype(I64),
                ]
            ),
            subject=plan[i, 0],
            channel_index=channel,
            group=view.n_dec,
            tick=jnp.stack([jnp.asarray(0, I64), following]),
            scope=Scope.CAMPAIGN,
            disposition=Disposition.DRAIN,
        )
        return eff, pstate, jnp.asarray(0.0, F64)


def survey_5(delay_ticks=catalog.CATALOG_DELAY_TICKS, policy=None, **overrides):
    """The five-acquisition survey whose catalog channel releases to everyone.

    `delay_ticks` is the catalog channel's release latency. `policy` defaults to the
    `FollowUpPolicy`; any other policy takes its place.
    """
    registry = toy.toy_registry()
    kind = catalog.register_catalog(registry)
    caps = registry.lower(Capacities(n_channels=4))
    overrides.setdefault("observer", RosterObserver(catalog_kind=kind))
    policy = FollowUpPolicy() if policy is None else policy
    base = builder.survey_1(caps=caps, policies=(policy,), **overrides)
    index = 3
    channels = base.channels
    channels = eqx.tree_at(
        lambda t: (t.kind, t.audience, t.latency_ticks),
        channels,
        (
            channels.kind.at[index].set(kind),
            channels.audience.at[index].set(PUBLIC),
            channels.latency_ticks.at[index].set(delay_ticks),
        ),
    )
    return dataclasses.replace(base, channels=channels)


def survey_5_registry():
    """The registry `survey_5`'s capacities are lowered from."""
    registry = toy.toy_registry()
    catalog.register_catalog(registry)
    return registry


__all__ = [
    "CANDIDATE_ID_BASE",
    "FOLLOW_CHANNEL_ID",
    "FOLLOW_SUBJECT",
    "LAST_CANDIDATE_SLOT",
    "ROSTER_COUNT_SLOT",
    "SURVEY_PLAN",
    "AbsolutePlanPolicy",
    "FollowUpPolicy",
    "RosterObserver",
    "survey_5",
    "survey_5_registry",
]
