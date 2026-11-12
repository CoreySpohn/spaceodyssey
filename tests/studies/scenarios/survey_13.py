"""A configuration switch: records carry the configuration their admission was under.

Asset 0 (lanes 0 and 1) declares one switch at `SWITCH_TICK`; asset 1 never switches.
`SwitchPolicy` observes subject 0 three times:

- an observation on lane 0 at tick 0, admitted before the switch (configuration 0);
- a commitment on lane 1 admitted at tick 0 that starts at `COMMIT_START`, after the
  switch, and so keeps configuration 0;
- a second observation on lane 0 at the decision that sees the first record, which is
  after the switch (configuration 1).

`ConfigPhotometry` is a version-2 channel model whose gain is the configuration's gain,
read from the stamp in the realize context, and whose row carries the stamp. A study
whose records were realized under another configuration than the stamped one shows it in
the value.
"""

import dataclasses

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.declarations import Declaration
from spaceodyssey.core.enums import (
    F64,
    I64,
    PUBLIC,
    Audience,
    Disposition,
    EffectKind,
    Scope,
)
from spaceodyssey.core.kinds import AbstractChannelModel, AbstractPolicy
from spaceodyssey.core.ledger import CONFIG_PAD
from spaceodyssey.core.records import KindRegistry, PayloadColumn, RecordKind
from spaceodyssey.studies import builder, effect_rows, toy

SWITCH_TICK = 100
COMMIT_START = 200
OBSERVE_CHANNEL, COMMIT_CHANNEL = 10, 11
GAINS = (1.0, 1.5)  # the gain of configurations 0 and 1


def survey_13_registry():
    """The toy kinds with the measurement row widened by the configuration stamp."""
    cols = (
        PayloadColumn("value", PUBLIC),
        PayloadColumn("variate", Audience.SCORER),
        PayloadColumn("fraction", PUBLIC),
        PayloadColumn("config", PUBLIC),
    )
    return KindRegistry(
        (RecordKind("none", ()), RecordKind("measurement", cols), toy.AUXILIARY_KIND)
    )


def survey_13_caps():
    """Commitments and a one-slot timeline on, with the widened registry lowered."""
    caps = Capacities(n_commitments=2, max_config_switches=1)
    return survey_13_registry().lower(caps)


class ConfigPhotometry(AbstractChannelModel):
    """The toy signal times the gain of the stamped configuration, plus noise.

    Version 2: the row is `[value, variate, fraction, config_id]`, and the gain is
    `gains[context.config_id]`.
    """

    noise: jnp.ndarray
    gains: jnp.ndarray
    version: int = eqx.field(default=2, kw_only=True)

    @classmethod
    def migrate(cls, from_version, params):
        """A version-1 document carries no field this model needs to change."""
        return params

    def realize(
        self, truth, subject, channel_index, fraction, key, design, channels, *, context
    ):
        """The measurement row under the stamped configuration."""
        z = jr.normal(key, dtype=F64)
        noise = self.noise[channel_index] / jnp.sqrt(jnp.maximum(fraction, 1e-3))
        value = self.gains[context.config_id] * truth.signal[subject] + noise * z
        return jnp.stack(
            [
                value,
                z,
                jnp.asarray(fraction, F64),
                jnp.asarray(context.config_id, F64),
            ]
        )


class SwitchPolicy(AbstractPolicy):
    """Observe, commit across the switch, and observe again after it; then drain.

    Decision 0 requests the observation on lane 0 and the commitment on lane 1. The
    first later decision (the one that sees a record) requests the second observation
    on lane 0, and the decision that sees the third record drains.
    """

    subject: int = eqx.field(default=0, static=True)

    def declare(self):
        """Observe and the closing campaign drain."""
        return Declaration(
            effects=frozenset((EffectKind.OBSERVE, EffectKind.STOP)),
            stops=((Scope.CAMPAIGN, Disposition.DRAIN),),
        )

    def decide(self, pstate, view, study, key):
        """Request the next step from the records seen and the observations made."""
        caps = study.caps
        seen = jnp.sum(view.belief.n)
        first = view.n_dec == 0
        again = ~first & (seen == 1) & (pstate[0] == 0)
        stop = seen >= 3
        eff = effect_rows.effects(
            caps,
            kind=jnp.stack(
                [
                    jnp.where(
                        first | again,
                        EffectKind.OBSERVE,
                        jnp.where(stop, EffectKind.STOP, EffectKind.NONE),
                    ),
                    jnp.where(first, EffectKind.OBSERVE, EffectKind.NONE),
                ]
            ).astype(I64),
            subject=self.subject,
            channel_index=jnp.stack(
                [
                    effect_rows.chan_index(study, OBSERVE_CHANNEL),
                    effect_rows.chan_index(study, COMMIT_CHANNEL),
                ]
            ),
            group=view.n_dec * caps.n_effects + jnp.arange(caps.n_effects),
            lane=jnp.stack([jnp.asarray(0, I64), jnp.asarray(1, I64)]),
            scope=Scope.CAMPAIGN,
            disposition=Disposition.DRAIN,
            start_tick=jnp.stack([jnp.asarray(0, I64), jnp.asarray(COMMIT_START, I64)]),
        )
        return eff, pstate.at[0].add(again.astype(F64)), jnp.asarray(0.0, F64)


def survey_13(switch_tick=SWITCH_TICK, **overrides):
    """The toy study with asset 0 switching at `switch_tick` and the switch policy."""
    caps = survey_13_caps()
    switches = jnp.asarray([[switch_tick], [CONFIG_PAD]], I64)
    overrides.setdefault("policies", (SwitchPolicy(),))
    study = builder.survey_1(caps, config_switch_ticks=switches, **overrides)
    model = ConfigPhotometry(
        noise=study.channel_model.noise, gains=jnp.asarray(GAINS, F64)
    )
    return dataclasses.replace(study, channel_model=model)


__all__ = [
    "COMMIT_CHANNEL",
    "COMMIT_START",
    "GAINS",
    "OBSERVE_CHANNEL",
    "SWITCH_TICK",
    "ConfigPhotometry",
    "SwitchPolicy",
    "survey_13",
    "survey_13_caps",
    "survey_13_registry",
]
