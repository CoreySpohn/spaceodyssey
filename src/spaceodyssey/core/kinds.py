"""Kinds."""

import dataclasses
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax

from spaceodyssey.core.enums import F64, I64, RecordStatus
from spaceodyssey.core.events import where_tree
from spaceodyssey.core.records import record_none
from spaceodyssey.core.snapshot import MeanBelief, mean_layout


@dataclasses.dataclass(frozen=True)
class ObserverCapabilities:
    """How an observer's belief changes: by incremental folds, by refits, or both.

    An `incremental` observer folds each record as it is accepted or released. A
    `refit` observer can recompute a subject's belief from that subject's rows; it
    names the record kinds it refits (`refit_kinds`, registered kind names, never a
    derived kind) and the statuses whose rows count (`refit_statuses`). For each
    acquisition the current revision is taken first and dropped when its status is
    not listed, so a correction can retract a value. `KindRegistry.lower` lowers the
    kinds and statuses into `Capacities`.
    """

    incremental: bool = True
    refit: bool = False
    refit_kinds: tuple[str, ...] = ()
    refit_statuses: tuple[int, ...] = (RecordStatus.COMPLETED, RecordStatus.PARTIAL)

    def __post_init__(self):
        """Refuse an observer that never changes, or refit kinds without a refit."""
        if not (self.incremental or self.refit):
            raise ValueError("an observer must be incremental, refit-capable or both")
        if bool(self.refit_kinds) != bool(self.refit):
            raise ValueError("refit_kinds must be nonempty exactly when refit is true")
        if len(set(self.refit_kinds)) != len(self.refit_kinds):
            raise ValueError("refit_kinds repeats a kind")
        statuses = tuple(RecordStatus(s) for s in self.refit_statuses)
        object.__setattr__(self, "refit_kinds", tuple(self.refit_kinds))
        object.__setattr__(self, "refit_statuses", statuses)


class AbstractObserver(eqx.Module):
    """Observer contract for record folding, refits and analysis.

    An observer keeps two belief types: the full belief (`Carry.belief`), built from
    the records the observer accepted, and the policy summary (`Carry.pbelief`),
    built from the records released to the policy and the only belief a policy
    reads. `layout` declares both types to the core and `capabilities` how they
    change.

    Full-belief methods (`fold`, `refit`, `install`) return the belief before
    analysis; the caller applies `analyze` to the whole belief after each change.
    Summary methods (`fold_summary`, `install_summary`) return the finished summary:
    summary analysis is not a separate step. `summary` maps a full belief to the
    summary type.

    A fold receives the key of the record it folds (`keys.fold_key`: the record's
    own identity under `Purpose.INFERENCE`), the same key for the belief and the
    summary; a refit receives the same per-row keys (`keys.fold_keys`) and,
    separately, the key of the subject's revision set. None is ever keyed by a
    count or an order.

    The defaults make the summary the full belief itself and the observer
    incremental only: `init_summary` is `init`, `summary` is the identity,
    `fold_summary` folds and analyzes, `install_summary` installs and analyzes, the
    layout is the `MeanBelief` layout of `init`, and `refit` folds the rows in
    order. `install` has no default: a refit-capable observer defines it.

    Promotion parity (`conformance.observer_kit`) holds an observer's refit to its
    incremental belief. An exact observer, the default, matches within the parity
    tolerances. A statistical observer, whose refit draws a different sample of the
    same posterior, sets the class attribute `statistical` and declares, also as
    class attributes, the equivalence `margin` of its trace projection, the `seeds`
    of its refit keys and the Monte Carlo `standard_error` the margin was sized
    against; the kit refuses a statistical observer without them.
    """

    statistical: ClassVar[bool] = False
    margin: ClassVar[float | None] = None
    seeds: ClassVar[tuple[int, ...]] = ()
    standard_error: ClassVar[float | None] = None

    version: int = eqx.field(default=1, kw_only=True)

    def capabilities(self):
        """The `ObserverCapabilities` of this observer: incremental only."""
        return ObserverCapabilities()

    def init(self, caps):
        """Initialize a fixed-shape belief state."""
        return MeanBelief(
            mean=jnp.zeros(caps.n_subjects, F64),
            n=jnp.zeros(caps.n_subjects, I64),
            aux=jnp.zeros(caps.n_an, F64),
        )

    def init_summary(self, caps):
        """Initialize the policy summary."""
        return self.init(caps)

    def layout(self, caps):
        """The `BeliefLayout` of this observer's belief and summary."""
        return mean_layout(
            jax.eval_shape(lambda: self.init(caps)),
            jax.eval_shape(lambda: self.init_summary(caps)),
        )

    def fold(self, belief, rec, caps, key=None):
        """Fold one record the observer accepted into the belief.

        `key` is the record's `Purpose.INFERENCE` key; an observer that draws
        nothing ignores it.
        """
        raise NotImplementedError

    def fold_summary(self, summary, rec, caps, key=None):
        """Fold a record released to the policy into the summary, then analyze it."""
        return self.analyze(self.fold(summary, rec, caps, key=key))

    def summary(self, belief):
        """The policy summary of a full belief."""
        return belief

    def refit(self, belief0, rows, caps, key, row_keys):
        """Recompute a belief from one subject's rows, starting from `belief0`.

        `rows` is a `Record` with a leading row axis, as `records.refit_rows` selects
        it: the subject's eligible rows in canonical order, masked for the audience,
        with `valid` false on every row that does not count (padding, a superseded
        revision, an ineligible kind or status). `row_keys` holds each row's fold
        key, `keys.fold_keys(root, rows, study)`, the keys the incremental folds of
        the same rows received. `key` is the refit key of the revision set, for an
        observer whose refit draws beyond its row folds.

        Returns a belief of `belief0`'s shape before the final analysis, like
        `fold`: the default folds each valid row in order, analyzing between folds as
        the incremental path does, so `analyze(refit(...))` of an observer's rows
        equals folding them one by one. A row that does not count is replaced by a
        row that does before its fold is evaluated and discarded, so padding cannot
        put a NaN into a gradient. With no valid row, `belief0` is returned.
        """
        first = jax.tree_util.tree_map(lambda x: x[jnp.argmax(rows.valid)], rows)
        stand_in = dataclasses.replace(first, valid=jnp.zeros_like(first.valid))

        def step(state, xs):
            belief, started = state
            row, row_key = xs
            safe = where_tree(row.valid, row, stand_in)
            base = where_tree(started, self.analyze(belief), belief)
            folded = self.fold(base, safe, caps, key=row_key)
            return (where_tree(row.valid, folded, belief), started | row.valid), None

        start = (belief0, jnp.zeros((), bool))
        return lax.scan(step, start, (rows, row_keys))[0][0]

    def install(self, belief, fitted, subject):
        """Replace `subject`'s part of `belief` with that of the refit belief `fitted`.

        Returns the belief before analysis.
        """
        raise NotImplementedError("a refit-capable observer defines install")

    def install_summary(self, summary, fitted, subject):
        """Replace `subject`'s part of `summary` with that of `fitted`, then analyze.

        `fitted` is `summary(refit result)`.
        """
        return self.analyze(self.install(summary, fitted, subject))

    def analyze(self, belief):
        """Analyze the whole full belief after a fold or an install."""
        return belief


class AbstractProcess(eqx.Module):
    """Contract for event seeding and process transitions."""

    version: int = eqx.field(default=1, kw_only=True)

    def seed(self, carry, caps):
        """Seed process events into the queue."""
        return carry

    def handle(self, carry, ev, study):
        """Handle a process event and return an optional record."""
        return carry, record_none(study.caps)


class AbstractPolicy(eqx.Module):
    """Policy contract for snapshot-based decisions."""

    version: int = eqx.field(default=1, kw_only=True)

    def decide(self, pstate, view, study, key):
        """Returns (EffectTable, new pstate, uniform draw u)."""
        raise NotImplementedError

    def decide_py(self, pstate, view_np, study, key):
        """Python-form policy for the host executor; same contract as `decide`."""
        return self.decide(pstate, view_np, study, key)


class AbstractWorldSource(eqx.Module):
    """Source of a world's latent state."""

    version: int = eqx.field(default=1, kw_only=True)

    def draw(self, key, capacities):
        """Draw one world's state."""
        raise NotImplementedError


class AbstractDesign(eqx.Module):
    """Continuous study design leaves."""

    version: int = eqx.field(default=1, kw_only=True)


class AbstractChannelModel(eqx.Module):
    """Realization contract for a channel's payload."""

    version: int = eqx.field(default=1, kw_only=True)

    def realize(self, truth, subject, channel_index, fraction, key, design, channels):
        """Return a realized payload and its random variate."""
        raise NotImplementedError


class AbstractObjective(eqx.Module):
    """Objective evaluated from the completed state."""

    version: int = eqx.field(default=1, kw_only=True)
    unit: str = eqx.field(default="subject", kw_only=True)

    def evaluate(self, carry, study):
        """Evaluate the objective."""
        raise NotImplementedError
