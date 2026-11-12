"""A frozen network choosing observations from released policy beliefs."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.declarations import Declaration
from spaceodyssey.core.enums import F64, I64, Disposition, EffectKind, Scope
from spaceodyssey.core.kinds import AbstractPolicy
from spaceodyssey.core.snapshot import effects_none
from spaceodyssey.studies.builder import survey_1


def _observation_limit(value):
    """Validate the observation limit before storing it as static configuration."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("n_obs must be a Python int; it is static configuration")
    if value < 0:
        raise ValueError("n_obs must be nonnegative")
    return value


class NetworkPolicy(AbstractPolicy):
    """Choose a subject/channel pair with an MLP, then drain after `n_obs` decisions.

    Inputs are the snapshot's per-subject means, per-subject sample counts,
    tick divided by the study horizon, and decision count divided by
    `max(n_obs, 1)`, in that order. Output logits are flattened in subject-major,
    channel-index-minor order. The largest logit wins, with the first index
    breaking ties. Only effect row zero is active.

    The network is frozen during a run. The observation limit is a deterministic
    stopping rule; it counts requests, including any admission refuses. Policy
    state is unchanged and no policy random draw is used.
    """

    network: eqx.nn.MLP
    n_obs: int = eqx.field(default=5, static=True, converter=_observation_limit)

    def declare(self):
        """Declare observation effects and the closing campaign drain."""
        return Declaration(
            effects=frozenset((EffectKind.OBSERVE, EffectKind.STOP)),
            stops=((Scope.CAMPAIGN, Disposition.DRAIN),),
        )

    def decide(self, pstate, view, study, key):
        """Evaluate the frozen network on the policy snapshot inside the scan."""
        caps = study.caps
        if self.network.in_size != 2 * caps.n_subjects + 2:
            raise ValueError("network input size must be 2 * n_subjects + 2")
        if self.network.out_size != caps.n_subjects * caps.n_channels:
            raise ValueError("network output size must be n_subjects * n_channels")
        features = jnp.concatenate(
            (
                view.belief.mean,
                view.belief.n.astype(F64),
                jnp.asarray(
                    [view.tick / study.horizon_tick, view.n_dec / max(self.n_obs, 1)],
                    F64,
                ),
            )
        )
        action = jnp.argmax(self.network(features)).astype(I64)
        subject, channel = action // caps.n_channels, action % caps.n_channels
        stop = view.n_dec >= self.n_obs
        effects = effects_none(caps)
        effects = dataclasses.replace(
            effects,
            kind=effects.kind.at[0].set(
                jnp.where(stop, EffectKind.STOP, EffectKind.OBSERVE)
            ),
            subject=effects.subject.at[0].set(jnp.where(stop, 0, subject)),
            channel_index=effects.channel_index.at[0].set(jnp.where(stop, 0, channel)),
            lane=effects.lane.at[0].set(
                jnp.where(stop, 0, study.channels.lane[channel])
            ),
            group=effects.group.at[0].set(view.n_dec),
            scope=effects.scope.at[0].set(jnp.where(stop, Scope.CAMPAIGN, 0)),
            disposition=effects.disposition.at[0].set(
                jnp.where(stop, Disposition.DRAIN, 0)
            ),
        )
        return effects, pstate, jnp.asarray(0.0, F64)


def survey_25(*, key=None, caps=None, n_obs=5, **overrides):
    """Build a toy study with a reproducibly initialized, frozen network policy.

    `key` initializes the network, independently of the executor's world roots.
    Omission uses key zero. The network has one eight-unit tanh hidden layer.
    Table, channel-model and horizon overrides follow `survey_1`.
    """
    caps = Capacities() if caps is None else caps
    key = jr.PRNGKey(0) if key is None else key
    network = eqx.nn.MLP(
        in_size=2 * caps.n_subjects + 2,
        out_size=caps.n_subjects * caps.n_channels,
        width_size=8,
        depth=1,
        activation=jax.nn.tanh,
        dtype=F64,
        key=key,
    )
    return survey_1(
        caps=caps, policies=(NetworkPolicy(network=network, n_obs=n_obs),), **overrides
    )


__all__ = ["NetworkPolicy", "survey_25"]
