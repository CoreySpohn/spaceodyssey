"""A scripted policy that starts activities later, and the toy study that holds them."""

import dataclasses

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from spaceodyssey.core import HostExecutor
from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.enums import F64, I64, ChargePricing
from tests.core.helpers import one_world
from tests.core.program_helpers import SCRIPT_ROWS, ScriptedPrograms

NONE = -1
HORIZON = 30  # the horizon of asset 0 (lanes 0 and 1); asset 1 (lane 2) has none
DISRUPTIVE_COL = 0  # the resource column an activation of asset 0 is debited from


def plan(n, slot, start, disruptive=0, grouped=0):
    """The start tick and flags of effect row `slot` of decision `n`.

    A row that is `grouped` joins the one group the decision's grouped rows share.
    """
    return (n, slot, start, int(disruptive), int(grouped))


def _as_plan(rows):
    out = np.full((SCRIPT_ROWS, 5), NONE, np.int64)
    for i, r in enumerate(rows):
        out[i, : len(r)] = r
    return jnp.asarray(out, I64)


class ScriptedCommitments(ScriptedPrograms):
    """The scripted effects, with the start tick and disruptive flag of `plan` rows."""

    plan: object = eqx.field(default=None, converter=lambda r: _as_plan(r or ()))

    def declare(self):
        """The script's declaration, disruptive when a row asks for it."""
        base = super().declare()
        return dataclasses.replace(base, disruptive=bool(np.any(self.plan[:, 3] > 0)))

    def decide(self, pstate, view, study, key):
        """Emit the script rows, then the start tick and flag of each planned row."""
        eff, ps, u = super().decide(pstate, view, study, key)
        n_effects = study.caps.n_effects
        start = jnp.zeros(n_effects, I64)
        flag = jnp.zeros(n_effects, jnp.bool_)
        group = eff.group
        for e in range(n_effects):
            pick = (self.plan[:, 0] == view.n_dec) & (self.plan[:, 1] == e)
            r = self.plan[jnp.argmax(pick)]
            start = start.at[e].set(jnp.where(pick.any(), r[2], 0))
            flag = flag.at[e].set(pick.any() & (r[3] > 0))
            joined = pick.any() & (r[4] > 0)
            group = group.at[e].set(jnp.where(joined, view.n_dec * n_effects, group[e]))
        eff = dataclasses.replace(eff, start_tick=start, disruptive=flag, group=group)
        return eff, ps, u


def commitment_study(
    script,
    plans,
    *,
    freeze=HORIZON,
    activations=1.0,
    n_programs=1,
    refunds=False,
    n_commitments=4,
    max_sid=None,
    **overrides,
):
    """The toy study with commitments on, a horizon on asset 0 and the scripted policy.

    Asset 0 holds `activations` of resource column `DISRUPTIVE_COL` beyond what its
    channels cost, and names that column as its disruptive resource.
    """
    from spaceodyssey.studies import toy

    extra = {} if max_sid is None else {"max_sid": max_sid}
    caps = Capacities(
        n_effects=3,
        n_commitments=n_commitments,
        n_programs=n_programs,
        refunds=refunds,
        **extra,
    )
    study = toy.survey_1(
        caps=caps,
        policies=(ScriptedCommitments(script=script, plan=plans),),
        **overrides,
    )
    assets = dataclasses.replace(
        study.assets,
        freeze_ticks=jnp.asarray([freeze, 0], I64),
        disruptive_resource=jnp.asarray([DISRUPTIVE_COL, -1], I64),
    )
    return dataclasses.replace(study, assets=assets)


def priced(study, pricing, charge, surcharge, balance=100.0):
    """The study with channel 10 priced `pricing` and every balance row at `balance`.

    Every study carries the pricing and surcharge columns, so one program serves all
    four pricing modes; the other channels keep a plain occupancy price.
    """
    ch = study.channels
    n_channels, n_resources = ch.cost.shape
    table = jnp.full((n_channels, n_resources), ChargePricing.OCCUPANCY, I64)
    channels = dataclasses.replace(
        ch,
        charge=ch.cost.at[0].set(jnp.asarray(charge, F64)),
        pricing=table.at[0].set(pricing),
        surcharge=jnp.zeros((n_channels, n_resources), F64)
        .at[0]
        .set(jnp.asarray(surcharge, F64)),
    )
    return dataclasses.replace(
        study, channels=channels, budget=jnp.full_like(study.budget, balance)
    )


def run(study, n_steps=200):
    """A host run of one world."""
    truth, root = one_world(caps=study.caps)
    return HostExecutor().run(study, truth, root, n_steps)
