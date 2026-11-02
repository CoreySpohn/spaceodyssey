"""Builders for the record tests: scripted records, spies and the leaking layout."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from spaceodyssey.core import (
    AbstractPolicy,
    Audience,
    KindRegistry,
)
from spaceodyssey.core.enums import F64, PUBLIC
from spaceodyssey.core.records import PayloadColumn, RecordKind
from spaceodyssey.studies import toy
from spaceodyssey.studies.scripted import ScriptedRecords as ScriptedRecords
from spaceodyssey.studies.scripted import scripted_study as scripted_study


def leaking_registry():
    """The old layout: the raw noise variate is a column every audience can read."""
    leaking = RecordKind(
        "measurement",
        (
            PayloadColumn("value", PUBLIC),
            PayloadColumn("variate", PUBLIC),
            PayloadColumn("fraction", PUBLIC),
        ),
    )
    return KindRegistry((RecordKind("none", ()), leaking, toy.AUXILIARY_KIND))


def with_leaking_layout(study):
    """The same study with the leaking layout lowered into its capacities."""
    caps = leaking_registry().lower(
        dataclasses.replace(study.caps, payload_audience=())
    )
    return dataclasses.replace(study, caps=caps)


def scorer_only_columns(caps):
    """bool[K, P]: columns only the scorer may read."""
    table = np.asarray(caps.payload_audience, dtype=np.int64)
    return table == int(Audience.SCORER)


SEEN = []


def _note(total):
    SEEN.append(float(total))


class SpyPolicy(AbstractPolicy):
    """Wraps a policy and reports the mass of `hidden` payload columns in its views.

    `hidden` is bool[K, P] over record kinds and columns, fixed when the spy is built so
    that a study with a different layout is judged against the same columns.
    """

    inner: AbstractPolicy
    hidden: tuple[tuple[bool, ...], ...] = eqx.field(static=True)

    def decide(self, pstate, view, study, key):
        """Report the hidden columns of the view, then defer to the inner policy."""
        hidden = jnp.asarray(self.hidden)
        kind = jnp.clip(view.log.kind, 0, hidden.shape[0] - 1)
        total = jnp.sum(jnp.abs(view.log.payload) * hidden[kind])
        jax.debug.callback(_note, total)
        return self.inner.decide(pstate, view, study, key)


def spied(study, reference=None):
    """The study with every policy wrapped in a `SpyPolicy`.

    The hidden columns are the scorer-only columns of `reference` (default: the
    study's own capacities).
    """
    mask = scorer_only_columns(study.caps if reference is None else reference)
    hidden = tuple(tuple(bool(x) for x in row) for row in mask)
    return eqx.tree_at(
        lambda s: s.agenda.policies,
        study,
        tuple(SpyPolicy(p, hidden) for p in study.agenda.policies),
    )


class ReconstructionPolicy(toy.FixedPlanPolicy):
    """Tries to recover the true signal from the first released row it sees.

    It inverts the channel model: signal = (value - noise * variate) / gain, which is
    exact when the raw variate is readable. The estimate and its subject (plus one) are
    kept in `pstate[2]` and `pstate[3]`.
    """

    def decide(self, pstate, view, study, key):
        """Plan as the fixed policy does, and record a reconstruction."""
        eff, ps, u = super().decide(pstate, view, study, key)
        row = view.log
        noise = study.channel_model.noise[row.channel_index[0]]
        est = (row.payload[0, 0] - noise * row.payload[0, 1]) / study.design.gain
        take = (view.n_rows > 0) & (ps[3] == 0)
        ps = ps.at[2].set(jnp.where(take, est, ps[2]))
        ps = ps.at[3].set(jnp.where(take, row.subject[0].astype(F64) + 1.0, ps[3]))
        return eff, ps, u
