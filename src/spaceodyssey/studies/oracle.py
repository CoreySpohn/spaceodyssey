"""A toy host service: it answers a policy's question from the snapshot it was asked on.

The service reads the policy belief in the snapshot, through the observer's layout,
and returns the projected mean and the count of the subject asked about, plus the
snapshot's release version. The answer is a record of its own kind, readable by the
policy audience only. Nothing in `spaceodyssey.core` names it.
"""

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.declarations import Declaration
from spaceodyssey.core.enums import F64, I64, Audience, EffectKind
from spaceodyssey.core.kinds import AbstractPolicy
from spaceodyssey.core.lift import traced
from spaceodyssey.core.service import AbstractHostService
from spaceodyssey.studies import builder, toy

ANSWER_KIND = toy.RecordKind(
    "answer",
    (
        toy.PayloadColumn("mean", Audience.POLICY),
        toy.PayloadColumn("count", Audience.POLICY),
        toy.PayloadColumn("version", Audience.POLICY),
    ),
)
DEFAULT_PLAN = ((0, 10), (1, 10), (0, 10), (1, 10), (0, 10))


class ToyOracleService(AbstractHostService):
    """Answer with the asked subject's belief as the snapshot shows it.

    The belief is read through the observer's `BeliefLayout`: the mean from the
    summary projection, the count from the count leaf.
    """

    def __init__(self, kind):
        """Answer with records of registry index `kind`."""
        self.kind = kind
        self.requests = []

    def answer(self, request, view, layout):
        """Return (mean, count, release version) for the subject of the request."""
        self.requests.append(request)
        subject = request.subject
        mean, _ = layout.summary_projection(view.belief)
        return np.asarray(
            [
                mean[subject],
                layout.count(view.belief)[subject],
                view.release_version,
            ],
            dtype=np.float64,
        )


class QueryPolicy(AbstractPolicy):
    """Follow a fixed plan, but ask the host about `subject` at decision `query_at`.

    The decision that asks does not advance the plan. The answer wakes the policy at
    the same tick, and the plan resumes on that decision.
    """

    inner: toy.FixedPlanPolicy
    query_at: int = eqx.field(default=1, metadata=traced(I64))
    subject: int = eqx.field(default=0, metadata=traced(I64))
    service: int = eqx.field(default=0, metadata=traced(I64))

    def declare(self):
        """The service channel the answer is recorded on, for pre-flight."""
        return Declaration(channels=(int(self.service),))

    def decide(self, pstate, view, study, key):
        """Ask at the chosen decision, otherwise act as the plan does."""
        eff, new_ps, u = self.inner.decide(pstate, view, study, key)
        ask = view.n_dec == self.query_at
        kind = eff.kind.at[0].set(jnp.where(ask, EffectKind.QUERY, eff.kind[0]))
        eff = eqx.tree_at(
            lambda e: (e.kind, e.subject, e.channel_index),
            eff,
            (
                kind,
                eff.subject.at[0].set(jnp.where(ask, self.subject, eff.subject[0])),
                eff.channel_index.at[0].set(
                    jnp.where(ask, self.service, eff.channel_index[0])
                ),
            ),
        )
        return eff, jnp.where(ask, pstate, new_ps), u.astype(F64)


def oracle_registry():
    """The toy registry with the answer kind added; returns it with the kind's index."""
    registry = toy.toy_registry()
    return registry, registry.register(ANSWER_KIND)


def oracle_study(query_at=1, subject=0, plan=DEFAULT_PLAN, **overrides):
    """A five-acquisition study whose policy asks the toy service once.

    Returns the study and the service that answers it.
    """
    registry, kind = oracle_registry()
    caps = registry.lower(overrides.pop("caps", Capacities()))
    policy = QueryPolicy(
        inner=toy.FixedPlanPolicy(plan=plan), query_at=query_at, subject=subject
    )
    study = builder.survey_1(caps=caps, policies=(policy,), **overrides)
    return study, ToyOracleService(kind)


__all__ = ["ANSWER_KIND", "QueryPolicy", "ToyOracleService", "oracle_study"]
