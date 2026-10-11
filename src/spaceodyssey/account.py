"""Accounting: the named-resource ledger and the reference cost model.

The ledger charges whatever named resources the mission specification
budgets; time ("time_d") is the universal resource, and further resources
(consumables such as retargeting fuel) are new ledger keys, not new
machinery. The cost-model protocol itself belongs to the scheduling
vocabulary (one instance shared by agent and environment); this module holds
the environment-side pieces: the ledger and the reference implementation.
"""

from typing import final

import equinox as eqx
import jax.numpy as jnp
from planit_py.vocabulary import (
    AbstractCostModel,
    AbstractIntegrationPolicy,
    Action,
    FixedDuration,
    ObservingContext,
    StopOnCertificate,
    StopOnInformationRate,
)

from spaceodyssey.registry import SeamInfo, register


@final
class Ledger(eqx.Module):
    """Immutable named-resource accounting.

    Attributes:
        budget: Resource caps, keyed by resource name.
        spent: Amount spent so far per resource.
    """

    budget: dict = eqx.field(default_factory=dict)
    spent: dict = eqx.field(default_factory=dict)

    @classmethod
    def from_budgets(cls, budgets: tuple[tuple[str, float], ...]) -> "Ledger":
        """Build a ledger from (resource, cap) pairs with nothing spent."""
        budget = dict(budgets)
        return cls(budget=budget, spent={name: 0.0 for name in budget})

    def charge(self, costs: dict) -> "Ledger":
        """Return a new ledger with the given costs added.

        Raises:
            KeyError: If a cost names a resource the budget does not declare;
                unbudgeted spending is a configuration error, not a warning.
        """
        spent = dict(self.spent)
        for resource, amount in costs.items():
            if resource not in self.budget:
                declared = ", ".join(sorted(self.budget)) or "(none)"
                raise KeyError(
                    f"cost charged to undeclared resource '{resource}'; "
                    f"budgeted resources: {declared}"
                )
            spent[resource] = spent.get(resource, 0.0) + float(amount)
        return Ledger(budget=dict(self.budget), spent=spent)

    def remaining(self, resource: str) -> float:
        """Return the unspent amount of one resource."""
        return self.budget[resource] - self.spent.get(resource, 0.0)

    @property
    def exhausted(self) -> bool:
        """True when any budgeted resource is fully spent."""
        return any(
            self.spent.get(resource, 0.0) >= cap
            for resource, cap in self.budget.items()
        )


def _integration_bound_d(policy: AbstractIntegrationPolicy) -> float:
    """The largest integration duration a policy can realize, in days."""
    if isinstance(policy, FixedDuration):
        return policy.duration_d
    if isinstance(policy, (StopOnCertificate, StopOnInformationRate)):
        return policy.max_duration_d
    raise TypeError(
        f"unknown integration policy {type(policy).__name__}; the cost model "
        "cannot bound its duration"
    )


def _slew_d(action: Action, ctx: ObservingContext) -> float:
    """Slew-plus-settle days to the action's target, from the context."""
    matches = jnp.nonzero(ctx.target_ids == action.target_id)[0]
    if matches.size == 0:
        raise KeyError(f"target {action.target_id} is not in the observing context")
    return float(ctx.slew_d[matches[0]])


@final
@register(
    "cost",
    "constant_overheads",
    SeamInfo(
        operators={"sample", "propagate", "integrate"},
        version="1",
        fidelity="constants",
        cost_hint="low",
    ),
)
class ConstantOverheads(AbstractCostModel):
    """Slew from the context plus a constant per-visit overhead.

    Expected costs bound adaptive integrations by their maximum duration;
    realized costs use the duration the environment reports.

    Attributes:
        overhead_d: Constant per-visit overhead in days (acquisition,
            settling, wavefront-control setup).
    """

    overhead_d: float = 0.0

    def expected(self, action: Action, ctx: ObservingContext) -> dict[str, float]:
        """Price an action before execution, bounding adaptive stopping."""
        integration_d = _integration_bound_d(action.integration)
        return {"time_d": integration_d + self.overhead_d + _slew_d(action, ctx)}

    def realized(
        self, action: Action, ctx: ObservingContext, duration_d: float
    ) -> dict[str, float]:
        """Price an executed action from its realized duration."""
        return {"time_d": duration_d + self.overhead_d + _slew_d(action, ctx)}
