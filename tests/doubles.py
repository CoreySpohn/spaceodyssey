"""Deterministic test doubles for the loop and engine tests.

The doubles exercise the seam machinery end to end without any physics: a
measurement whose detection statistic depends only on whether the truth
scene holds a planet, a round-robin policy, and a fixed three-target
universe (planets at targets 1 and 3, nothing at target 2).
"""

from dataclasses import replace
from typing import final

import jax.numpy as jnp
from planit_py.vocabulary import (
    AbstractPolicy,
    Action,
    FixedDuration,
    Mode,
)

from spaceodyssey.account import ConstantOverheads, Ledger
from spaceodyssey.certify import SnrThreshold
from spaceodyssey.context import AlwaysObservable
from spaceodyssey.data import SummaryDataset
from spaceodyssey.dist import AbstractPredictive
from spaceodyssey.errors import UnsupportedOperator
from spaceodyssey.observe import AbstractMeasurement
from spaceodyssey.registry import Registry, SeamInfo
from spaceodyssey.spec import FidelityConfig, MissionSpec, SeamRef
from spaceodyssey.universe import AbstractUniverseSource, Universe

TARGET_IDS = (1, 2, 3)
PLANET_TARGETS = (1, 3)


class FirstImplementation:
    """A simple implementation whose class identity can be checked."""

    def __init__(self, value=0):
        """Retain a constructor parameter."""
        self.value = value


class SecondImplementation(FirstImplementation):
    """An alternative class with the same constructor signature."""


class ChangingRegistry(Registry):
    """Return alternating classes to expose repeated reference resolution."""

    def __init__(self):
        """Register a simple scene implementation and count lookups."""
        super().__init__()
        self.calls = []
        self.register("scene", "changing", SeamInfo(operators={"sample"}))(
            FirstImplementation
        )

    def get(self, seam, name):
        """Return a different implementation class on each lookup."""
        self.calls.append((seam, name))
        entry = super().get(seam, name)
        cls = FirstImplementation if len(self.calls) % 2 else SecondImplementation
        return replace(entry, cls=cls)


@final
class DeltaDataset(AbstractPredictive):
    """A predictive concentrated on one fixed dataset."""

    dataset: SummaryDataset

    def sample(self, key):
        """Return the fixed dataset regardless of key."""
        return self.dataset

    def moments(self):
        """Raise: a dataset-valued delta exposes no numeric moments."""
        raise UnsupportedOperator("delta predictive has no moments")

    def integrate(self, functional):
        """Raise: not needed by the sampling path."""
        raise UnsupportedOperator("delta predictive does not integrate")

    def log_prob(self, value):
        """Raise: not needed by the sampling path."""
        raise UnsupportedOperator("delta predictive has no density")


@final
class PresenceMeasurement(AbstractMeasurement):
    """Detection statistic set by truth-scene presence, nothing else.

    Attributes:
        snr_present: Statistic reported when the scene holds a source.
        snr_absent: Statistic reported when it does not.
    """

    snr_present: float = 9.0
    snr_absent: float = 1.0

    def predictive(self, scene, action):
        """Return a delta on the dataset this scene deterministically yields."""
        snr = self.snr_present if scene else self.snr_absent
        dataset = SummaryDataset(
            values=jnp.asarray([0.0]),
            sigma=jnp.asarray([1.0]),
            snr=snr,
            epoch_d=action.start_time_d,
            labels=("x_arcsec",),
            band=action.bands[0] if action.bands else "",
        )
        return DeltaDataset(dataset=dataset)


@final
class RoundRobin(AbstractPolicy):
    """Visit the context's targets in order, one fixed-length look each.

    Stateless: the next index derives from the total number of observations
    already in the belief.

    Attributes:
        duration_d: Integration duration of every proposed action, in days.
    """

    duration_d: float = 1.0

    def propose(self, belief, ctx, cost_model):
        """Propose the next target in cyclic order."""
        total = sum(len(target.observations) for target in belief.per_target.values())
        index = total % ctx.target_ids.shape[0]
        return Action(
            target_id=int(ctx.target_ids[index]),
            mode=Mode.IMAGING,
            bands=("500nm",),
            integration=FixedDuration(duration_d=self.duration_d),
            start_time_d=ctx.time_d,
        )


@final
class ThreeTargetSource(AbstractUniverseSource):
    """A fixed universe: planets at targets 1 and 3, nothing at target 2."""

    def draw(self, key):
        """Return the fixed three-target universe."""
        return Universe(
            systems={1: "planet", 2: None, 3: "planet"},
            exozodi=dict.fromkeys(TARGET_IDS, 1.0),
        )


def build_test_registry() -> Registry:
    """A registry wiring the doubles plus the real reference implementations."""
    registry = Registry()
    registry.register(
        "scene",
        "three_targets",
        SeamInfo(operators={"sample"}, produces=("spaceodyssey.Universe",)),
    )(ThreeTargetSource)
    registry.register(
        "observation",
        "presence",
        SeamInfo(operators={"sample"}, produces=("spaceodyssey.SummaryDataset",)),
    )(PresenceMeasurement)
    registry.register(
        "certification",
        "snr_threshold",
        SeamInfo(
            operators={"sample"},
            consumes=("spaceodyssey.SummaryDataset",),
            produces=("spaceodyssey.CertificateState",),
        ),
    )(SnrThreshold)
    registry.register(
        "policy",
        "round_robin",
        SeamInfo(operators={"sample"}),
    )(RoundRobin)
    registry.register(
        "context",
        "always_observable",
        SeamInfo(
            operators={"sample"},
            produces=("planit_py.ObservingContext",),
        ),
    )(AlwaysObservable)
    registry.register(
        "cost",
        "constant_overheads",
        SeamInfo(operators={"sample"}),
    )(ConstantOverheads)
    return registry


def make_spec(duration_d: float = 6.0, time_budget_d: float = 100.0) -> MissionSpec:
    """A three-target mission spec with a time budget."""
    return MissionSpec.make(
        "toy-mission",
        duration_d,
        budgets={"time_d": time_budget_d},
        bands=("500nm",),
    )


def make_config(
    threshold: float = 7.0,
    overhead_d: float = 0.1,
    policy_duration_d: float = 1.0,
    snr_absent: float = 1.0,
) -> FidelityConfig:
    """A sample-operator configuration over the doubles."""
    return FidelityConfig.make(
        "sample",
        scene=SeamRef.make("scene", "three_targets"),
        observation=SeamRef.make("observation", "presence", snr_absent=snr_absent),
        certification=SeamRef.make(
            "certification", "snr_threshold", threshold=threshold
        ),
        policy=SeamRef.make("policy", "round_robin", duration_d=policy_duration_d),
        context=SeamRef.make("context", "always_observable", target_ids=TARGET_IDS),
        cost=SeamRef.make("cost", "constant_overheads", overhead_d=overhead_d),
    )


def make_ledger(time_budget_d: float = 100.0) -> Ledger:
    """A single-resource ledger for direct loop tests."""
    return Ledger.from_budgets((("time_d", time_budget_d),))
