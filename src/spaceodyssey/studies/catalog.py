"""A delayed catalog row released to the observer and never to the policy.

The catalog kind registers itself through `KindRegistry.register`; nothing in
`spaceodyssey.core` names it.
"""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp

from spaceodyssey.core.capacities import PLAIN_VALUE, Capacities
from spaceodyssey.core.enums import I64, Audience
from spaceodyssey.core.snapshot import MeanBelief
from spaceodyssey.plain import plain_array
from spaceodyssey.studies import toy

CATALOG_CHANNEL_ID = 13
CATALOG_DELAY_TICKS = 150
CATALOG_AUX_SLOT = 1

CATALOG_KIND = toy.RecordKind(
    "catalog",
    (
        toy.PayloadColumn("value", Audience.OBSERVER),
        toy.PayloadColumn("variate", Audience.SCORER),
        toy.PayloadColumn("fraction", Audience.OBSERVER),
    ),
)


def register_catalog(registry):
    """Add the catalog kind to `registry` and return its record-kind index."""
    return registry.register(CATALOG_KIND)


def _as_i64(x):
    return plain_array(I64, x)


class CatalogObserver(toy.ToyRunningMean):
    """Running mean of measurements; catalog values accumulate in an auxiliary slot."""

    catalog_kind: jax.Array = eqx.field(
        default=3, converter=_as_i64, metadata=PLAIN_VALUE
    )

    def fold(self, belief, rec, caps, key=None):
        """Fold as the running mean does, then add a catalog row's value."""
        belief = super().fold(belief, rec, caps, key=key)
        row = rec.valid & (rec.kind == self.catalog_kind)
        aux = belief.aux.at[CATALOG_AUX_SLOT].add(jnp.where(row, rec.payload[0], 0.0))
        return MeanBelief(belief.mean, belief.n, aux)


def catalog_study(**overrides):
    """A five-acquisition study whose fourth channel logs catalog rows.

    The catalog channel releases its row `CATALOG_DELAY_TICKS` after the occupancy
    ends, to the observer only.
    """
    registry = toy.toy_registry()
    kind = register_catalog(registry)
    caps = registry.lower(Capacities(n_channels=4))
    overrides.setdefault("observer", CatalogObserver(catalog_kind=kind))
    base = toy.survey_1(caps=caps, **overrides)
    index = 3
    channels = base.channels
    channels = eqx.tree_at(
        lambda t: (t.kind, t.audience, t.latency_ticks),
        channels,
        (
            channels.kind.at[index].set(kind),
            channels.audience.at[index].set(Audience.OBSERVER),
            channels.latency_ticks.at[index].set(CATALOG_DELAY_TICKS),
        ),
    )
    return dataclasses.replace(base, channels=channels)


__all__ = [
    "CATALOG_CHANNEL_ID",
    "CATALOG_DELAY_TICKS",
    "CATALOG_KIND",
    "CatalogObserver",
    "catalog_study",
    "register_catalog",
]
