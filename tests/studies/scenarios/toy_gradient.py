"""The toy gradient experiment: replayed policy, forced effects, smooth credit."""

import jax
import jax.numpy as jnp

from spaceodyssey.core.enums import (
    F64,
    I64,
    Disposition,
    EffectKind,
    Scope,
)
from spaceodyssey.core.kinds import (
    AbstractPolicy,
)
from spaceodyssey.core.snapshot import (
    EffectTable,
)
from spaceodyssey.studies.builder import survey_1
from spaceodyssey.studies.effect_rows import chan_index, effects
from spaceodyssey.studies.toy import (
    StochasticPolicy,
    carry_snr,
)


class ReplayPolicy(AbstractPolicy):
    """Emit recorded effects by decision index, repeating the last row.

    Each field of effects has shape (maximum_decisions, maximum_effects). Wait ticks are
    absolute, so replay freezes the discrete path.
    """

    effects: EffectTable

    def decide(self, pstate, view, study, key):
        """Return the recorded effects for the current decision."""
        n_max = self.effects.kind.shape[0]
        i = jnp.minimum(view.n_dec, n_max - 1)
        eff = jax.tree_util.tree_map(lambda a: a[i], self.effects)
        return eff, pstate, jnp.asarray(0.0, F64)


def forced_effects(caps, study, m, n_dec_max=6):
    """Build m alternating-subject observations followed by a campaign drain."""
    rows = []
    for j in range(n_dec_max):
        if j < m:
            k0, subject = EffectKind.OBSERVE, j % 2
        else:
            k0, subject = EffectKind.STOP, 0
        rows.append(
            effects(
                caps,
                kind=jnp.stack(
                    [jnp.asarray(k0, I64), jnp.asarray(EffectKind.NONE, I64)]
                ),
                subject=subject,
                channel_index=chan_index(study, 10),
                group=j,
                scope=Scope.CAMPAIGN,
                disposition=Disposition.DRAIN,
            )
        )
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *rows)


def credit_soft(carry, study, width):
    """Smooth stand-in for `credit_hard`: sum of sigmoid((snr - thr) / width)."""
    return jnp.sum(
        jax.nn.sigmoid((carry_snr(carry, study) - study.objective.threshold) / width)
    )


def d2_study(theta=0.3, **overrides):
    """Build a study with five stochastic observe-or-wait decisions."""
    return survey_1(policies=(StochasticPolicy(theta=theta),), **overrides)
