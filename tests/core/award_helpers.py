"""A scripted policy that allocates awards, and the toy study that holds them."""

import dataclasses

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from spaceodyssey.core import HostExecutor
from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.enums import F64, I64, EffectKind, RemainderRule
from tests.core.helpers import one_world
from tests.core.program_helpers import SCRIPT_ROWS, ScriptedPrograms, row

NONE = -1
N_RESOURCES = 3
PROGRAM_COL = 1  # the resource whose scope is the program (asset, program, shared)


def award(n, slot, recipient, start, until, amount, rule=RemainderRule.NONE, to=0):
    """An Allocate in effect row `slot` of decision `n`: `amount` (one entry per
    resource, or the program column alone) to balance row `recipient` for the window
    `[start, until)`, with `rule` at its end (`to` is a REALLOCATE's destination)."""
    if np.ndim(amount) == 0:
        amount = tuple(float(amount) if k == PROGRAM_COL else 0.0 for k in range(3))
    return (n, slot, recipient, start, until, *amount, int(rule), to)


def _as_awards(rows):
    out = np.full((SCRIPT_ROWS, 7 + N_RESOURCES), float(NONE), np.float64)
    for i, r in enumerate(rows):
        out[i] = r
    return jnp.asarray(out, F64)


class ScriptedAwards(ScriptedPrograms):
    """The scripted effects, plus the Allocate rows of `awards` (see `award`)."""

    awards: object = eqx.field(default=None, converter=lambda r: _as_awards(r or ()))

    def declare(self):
        """The programs of the script, and the Allocate effect."""
        base = super().declare()
        return dataclasses.replace(base, effects=frozenset({EffectKind.ALLOCATE}))

    def decide(self, pstate, view, study, key):
        """Emit the script rows, then overwrite each named slot with its Allocate."""
        eff, ps, u = super().decide(pstate, view, study, key)
        caps = study.caps
        a = self.awards
        kind = eff.kind
        amount = jnp.zeros((caps.n_effects, caps.n_resources), F64)
        cols = {k: jnp.zeros(caps.n_effects, I64) for k in ("from", "until", "to")}
        recipient = rule = jnp.zeros(caps.n_effects, I64)
        for e in range(caps.n_effects):
            pick = (a[:, 0] == view.n_dec) & (a[:, 1] == e)
            has = pick.any()
            r = a[jnp.argmax(pick)]
            kind = kind.at[e].set(jnp.where(has, EffectKind.ALLOCATE, kind[e]))
            amount = amount.at[e].set(jnp.where(has, r[5 : 5 + caps.n_resources], 0.0))
            recipient = recipient.at[e].set(jnp.where(has, r[2], 0).astype(I64))
            cols["from"] = cols["from"].at[e].set(jnp.where(has, r[3], 0).astype(I64))
            cols["until"] = cols["until"].at[e].set(jnp.where(has, r[4], 0).astype(I64))
            rule = rule.at[e].set(jnp.where(has, r[5 + N_RESOURCES], 0).astype(I64))
            cols["to"] = (
                cols["to"].at[e].set(jnp.where(has, r[6 + N_RESOURCES], 0).astype(I64))
            )
        eff = dataclasses.replace(
            eff,
            kind=kind,
            amount=amount,
            window_from=cols["from"],
            window_until=cols["until"],
            recipient=recipient,
            remainder=rule,
            realloc_to=cols["to"],
        )
        return eff, ps, u


def award_study(
    script,
    awards,
    *,
    program_credit=0.0,
    refunds=False,
    policy_cls=None,
    caps_extra=None,
    **overrides,
):
    """The toy study with two programs and awards on; the program rows hold
    `program_credit` of the program-scoped resource, so awards are what pay."""
    from spaceodyssey.studies import toy

    caps = Capacities(
        n_programs=2, n_effects=3, awards=True, refunds=refunds, **(caps_extra or {})
    )
    policy = (policy_cls or ScriptedAwards)(script=script, awards=awards)
    study = toy.survey_1(caps=caps, policies=(policy,), **overrides)
    budget = study.budget
    for p in range(caps.n_programs):
        budget = budget.at[caps.n_assets + p, PROGRAM_COL].set(program_credit)
    return dataclasses.replace(study, budget=budget)


def run(study, n_steps=200):
    """A host run of one world."""
    truth, root = one_world(caps=study.caps)
    return HostExecutor().run(study, truth, root, n_steps)


def award_refund_study(script, awards, receipts=(), **overrides):
    """The refund study (every row holds 100, channels cost 10 of each resource) with
    two programs, awards on, the scripted awards and the scripted receipts."""
    from tests.core.refund_helpers import refund_study

    return refund_study(
        script,
        receipts,
        cap_kw={"n_programs": 2, "awards": True},
        policy=ScriptedAwards(script=script, awards=awards),
        **overrides,
    )


__all__ = ["award", "award_refund_study", "award_study", "row", "run"]
