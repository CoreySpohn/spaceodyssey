"""Leaf types shared by every layer: draws, pool, rows, actions, the model contract.

This module imports nothing from the package. Axis names used in the annotations:
``A`` draws, ``K`` planets one draw can hold, ``G`` gases, ``R`` roles (forecast is
row 0, the verdict members are rows 1 to ``R - 1``), ``W`` log row capacity and ``Am``
the atoms of a pool.
"""

import abc
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray, PyTree


def _asarrays(tree):
    """Convert every leaf of ``tree`` to a JAX array."""
    return jax.tree.map(jnp.asarray, tree)


class WeightedCloud(eqx.Module):
    """Weighted draws from one refit; a stand-in mirroring the refit cloud's fields.

    ``samples`` is a pytree whose leaves share the leading draw axis ``A``.

    Weight contract: ``log_weight`` is the log of target over proposal, where the
    target is ``p0`` times the rows in ``row_mask``. It is not a log ratio against a
    proposal that equals ``p0``: that reading would count ``-log p0`` twice when the
    role weights are formed. One cloud has one row set, so ``row_mask`` is one mask
    for all its draws; a block normalization would need one per draw. The cloud comes
    from one refit.

    Attributes:
        samples: the state pytree, leading axis ``A``.
        log_weight: log of target over proposal, shape ``(A,)``.
        log_ref_density: log of ``p0`` on ``measure``, shape ``(A,)``.
        measure: the coordinates in which densities are taken, a static label.
    """

    samples: PyTree = eqx.field(converter=_asarrays)
    log_weight: Float[Array, " A"] = eqx.field(converter=jnp.asarray)
    log_ref_density: Float[Array, " A"] = eqx.field(converter=jnp.asarray)
    measure: str = eqx.field(static=True)


class Atoms[StateT](eqx.Module):
    """A weighted cloud with the gas block and the rows the cloud already represents.

    ``cloud.samples`` carries the state of type ``StateT``. The gas block holds, per
    draw, planet and gas, a Gaussian over the abundance: ``x_mean`` and ``x_var`` (zero
    variance is a point state).

    ``K`` is the number of planets one draw can hold; a draw with fewer is padded, and
    ``exists`` marks its real slots. A slot that does not exist is claim-false whatever
    its state, so padding a draw changes no probability. Its state is masked, not
    read, in the values; keep it finite all the same (zeros, say), because a NaN in a
    masked branch still turns a derivative into NaN. A draw padded only to give every
    star one shape carries zero weight (``log_weight = -inf``).

    Attributes:
        cloud: the weighted draws.
        x_mean: mean abundance, shape ``(A, K, G)``.
        x_var: abundance variance, shape ``(A, K, G)``, zero for a point state.
        row_mask: log rows the refit conditioned on, shape ``(W,)``, one set for the
            whole cloud.
        exists: whether each planet slot of each draw holds a planet, shape ``(A, K)``.
    """

    cloud: WeightedCloud
    x_mean: Float[Array, "A K G"] = eqx.field(converter=jnp.asarray)
    x_var: Float[Array, "A K G"] = eqx.field(converter=jnp.asarray)
    row_mask: Bool[Array, " W"] = eqx.field(converter=jnp.asarray)
    exists: Bool[Array, "A K"] = eqx.field(converter=jnp.asarray)


class Pool(eqx.Module):
    """The undetected planets at a star, per role.

    Each mark is one planet (``K = 1`` in ``marks``); the marks carry the conditional
    distribution of an undetected planet and the pool carries how many there are in
    expectation. A mark's weight under role ``r`` is formed like a draw's,
    ``log_weight + log_prior[r] - log_ref_density + log_like[r]``, normalized over the
    marks.

    The pool keeps its state at time zero beside the current one. A null observation
    thins the pool: it lowers ``log_mass`` and adds the log probability of not being
    detected to ``log_like``. ``log_mass0`` and ``log_like0`` stay, so the claim-true
    rate at time zero, ``exp(log_mass0) * pbar0`` with ``pbar0`` under the time-zero
    mark weights, can be set against the current one. A grant enters at time zero, so
    its rows are already in ``log_like0``.

    Attributes:
        marks: the marks, leading axis ``Am``.
        log_prior: log prior of each mark under each role, shape ``(R, Am)``.
        log_like: log likelihood of each mark under each role, shape ``(R, Am)``.
        log_mass: log expected number of planets, per role, shape ``(R,)``.
        log_mass0: the same at time zero, shape ``(R,)``.
        log_like0: ``log_like`` at time zero, shape ``(R, Am)``.

    Raises:
        ValueError: if a mark holds more than one planet, or the fields disagree on
            the number of roles or marks.
    """

    marks: Atoms
    log_prior: Float[Array, "R Am"] = eqx.field(converter=jnp.asarray)
    log_like: Float[Array, "R Am"] = eqx.field(converter=jnp.asarray)
    log_mass: Float[Array, " R"] = eqx.field(converter=jnp.asarray)
    log_mass0: Float[Array, " R"] = eqx.field(converter=jnp.asarray)
    log_like0: Float[Array, "R Am"] = eqx.field(converter=jnp.asarray)

    def __check_init__(self):
        """Check the static shapes: one planet per mark, one ``R`` and one ``Am``."""
        if self.marks.x_mean.shape[-2] != 1:
            raise ValueError(
                f"a mark is one planet (K = 1), not K = {self.marks.x_mean.shape[-2]}"
            )
        n_mark = self.marks.cloud.log_weight.shape[-1]
        per_mark = (self.log_prior, self.log_like, self.log_like0)
        roles = {x.shape[-2] for x in per_mark}
        roles |= {x.shape[-1] for x in (self.log_mass, self.log_mass0)}
        if len(roles) != 1 or {x.shape[-1] for x in per_mark} != {n_mark}:
            raise ValueError(
                f"pool fields disagree on R or Am (log_prior {self.log_prior.shape}, "
                f"log_like {self.log_like.shape}, log_like0 {self.log_like0.shape}, "
                f"log_mass {self.log_mass.shape}, log_mass0 {self.log_mass0.shape}, "
                f"{n_mark} marks)"
            )


class Row(eqx.Module):
    """One released observation result.

    The log is a ``Row`` stacked along a leading axis (``ObservationLog``), so its
    columns are these fields.

    Attributes:
        star: star index.
        mode: observing mode index.
        time_d: start time, days from the start of the survey.
        exposure_d: exposure length in days.
        charged_d: time charged against the budget in days.
        release_d: time at which the row is released, in days.
        refusal: refusal code, zero for an accepted row.
        model: index of the model that inverts the payload.
        payload: one payload per model, each a dict of named arrays, zero-padded; only
            the payload of ``model`` is meaningful.
    """

    star: Int[Array, ""] = eqx.field(converter=jnp.asarray)
    mode: Int[Array, ""] = eqx.field(converter=jnp.asarray)
    time_d: Float[Array, ""] = eqx.field(converter=jnp.asarray)
    exposure_d: Float[Array, ""] = eqx.field(converter=jnp.asarray)
    charged_d: Float[Array, ""] = eqx.field(converter=jnp.asarray)
    release_d: Float[Array, ""] = eqx.field(converter=jnp.asarray)
    refusal: Int[Array, ""] = eqx.field(converter=jnp.asarray)
    model: Int[Array, ""] = eqx.field(converter=jnp.asarray)
    payload: tuple[PyTree, ...] = eqx.field(
        converter=lambda p: tuple(_asarrays(x) for x in p)
    )


class ObservationLog(eqx.Module):
    """The rows of one run, at a fixed capacity ``W``.

    Row ``i`` is real when ``i < count``; the rows past ``count`` are padding, zeros.
    Grant rows come first (rows ``0`` to ``G - 1``), then the observed rows.

    Attributes:
        rows: a ``Row`` whose leaves have leading axis ``W``.
        count: the number of real rows.

    Raises:
        ValueError: if the leaves of ``rows`` disagree on ``W``.
    """

    rows: Row
    count: Int[Array, ""] = eqx.field(converter=jnp.asarray)

    def __check_init__(self):
        """Check the static shapes: one capacity ``W`` across the columns."""
        capacity = {jnp.shape(x)[0] for x in jax.tree.leaves(self.rows)}
        if len(capacity) != 1:
            raise ValueError(f"log columns disagree on the capacity W: {capacity}")


class Action(eqx.Module):
    """What a policy asks for: one pytree with an integer kind.

    Attributes:
        kind: integer action kind.
        star: star index.
        mode: observing mode index.
        exposure_d: requested exposure in days.
    """

    kind: Int[Array, ""] = eqx.field(converter=jnp.asarray)
    star: Int[Array, ""] = eqx.field(converter=jnp.asarray)
    mode: Int[Array, ""] = eqx.field(converter=jnp.asarray)
    exposure_d: Float[Array, ""] = eqx.field(converter=jnp.asarray)


class Grant(eqx.Module):
    """The actions a grant reveals at time zero, stacked along a leading axis.

    A grant holds no data: the world draws each granted row from its truth, as for an
    observation, and releases it at time zero, charged nothing. Granted row ``i`` is
    row ``i`` of the log.

    Attributes:
        actions: an ``Action`` whose leaves have a leading axis over the granted rows;
            each gives the star, the mode and the exposure that sets the S/N.
        model: the model slot of each granted row, static, one entry per row.
    """

    actions: Action
    model: tuple[int, ...] = eqx.field(
        static=True, converter=lambda s: tuple(int(i) for i in s)
    )

    def __check_init__(self):
        """Check that every granted row has one model slot."""
        if jnp.shape(self.actions.star) != (len(self.model),):
            raise ValueError("a grant has one model slot per granted action")


def grant_rows(grant: Grant, payload: tuple[PyTree, ...]) -> Row:
    """The log rows of a grant: released at time zero and charged nothing.

    Each row keeps its action's star, mode and exposure and carries its model slot.

    Args:
        grant: the grant.
        payload: one payload buffer per model, leading axis over the granted rows, as
            ``SealedWorld.reveal`` returns them.

    Returns:
        The granted rows, stacked along a leading axis.
    """
    actions = grant.actions
    zero_d = jnp.zeros_like(actions.exposure_d)
    zero_i = jnp.zeros_like(actions.star)
    return Row(
        star=actions.star,
        mode=actions.mode,
        time_d=zero_d,
        exposure_d=actions.exposure_d,
        charged_d=zero_d,
        release_d=zero_d,
        refusal=zero_i,
        model=jnp.asarray(grant.model, dtype=zero_i.dtype).reshape(zero_i.shape),
        payload=payload,
    )


class AbstractObservationModel(eqx.Module):
    """The contract of an observation model, a row's payload given a draw and an action.

    A model holds configuration only, never an observed datum; it is handed a payload
    each time it scores one. That is a deliberate difference from a data-bearing
    likelihood term: the log is a fixed-capacity buffer of payloads per model, scored
    by one commutative sum per model.

    A row is scored through additive sufficient statistics. ``stats`` maps one row to
    per-draw statistics; the statistics of a model's rows are summed (leaf by leaf, in
    any order); ``finalize`` turns the sum into the joint log marginal likelihood per
    draw and the updated draws. The per-row log likelihood and the update by one row
    are derived from this pair (``row_log_like``, ``row_update``), so there is one
    scoring path.

    For a model whose rows are independent given the draw, ``stats`` is the per-row log
    likelihood and ``finalize`` returns the summed statistics with the draws unchanged.
    A model whose draws carry a distribution that several rows read (a Gaussian gas
    block with ``x_var > 0``) cannot use that form, since the rows are not independent
    given the draw; it carries conjugate statistics instead.
    """

    @abc.abstractmethod
    def sample(self, atoms: Atoms, action: Action, key: PRNGKeyArray) -> PyTree:
        """Draw the payload an action would return, independently for each draw.

        Args:
            atoms: the draws.
            action: the action.
            key: the key of the row.

        Returns:
            The payload, a dict of named arrays with leading axis ``A``.
        """

    @abc.abstractmethod
    def stats(self, payload: PyTree, atoms: Atoms, star) -> PyTree:
        """Additive sufficient statistics of one row under each draw.

        Args:
            payload: the payload of one row, a dict of named arrays.
            atoms: the draws.
            star: a pytree of scalar leaves describing the star.

        Returns:
            A pytree whose leaves have leading axis ``A``; the statistics of several
            rows are their leafwise sum.
        """

    @abc.abstractmethod
    def finalize(
        self, stats: PyTree, atoms: Atoms, star
    ) -> tuple[Float[Array, " A"], Atoms]:
        """Joint log marginal likelihood and updated draws from summed statistics.

        Args:
            stats: the leafwise sum of ``stats`` over the rows of this model.
            atoms: the draws the rows are scored against.
            star: a pytree of scalar leaves describing the star.

        Returns:
            The log marginal likelihood of the rows per draw, shape ``(A,)``, and the
            draws updated by the rows.
        """

    @abc.abstractmethod
    def cells(self, atoms: Atoms, action: Action) -> Float[Array, "A Y"]:
        """Probability of each outcome cell under each draw.

        A model without a closed form returns ``binned_cells``.

        Args:
            atoms: the draws.
            action: the action.

        Returns:
            Cell probabilities of shape ``(A, Y)``; each row sums to one.
        """


def row_log_like(
    model: AbstractObservationModel, payload: PyTree, atoms: Atoms, star
) -> Float[Array, " A"]:
    """Log likelihood of one row under each draw: ``finalize`` of its ``stats``.

    For a draw with a distribution over the quantity read (``x_var > 0``) this is the
    row's marginal; the marginals of several rows reading one quantity do not sum to
    their joint likelihood. Sum the statistics instead.

    Args:
        model: the row's observation model.
        payload: the payload of the row.
        atoms: the draws.
        star: a pytree of scalar leaves describing the star.

    Returns:
        Log likelihood per draw, shape ``(A,)``.
    """
    return model.finalize(model.stats(payload, atoms, star), atoms, star)[0]


def row_update(
    model: AbstractObservationModel, payload: PyTree, atoms: Atoms, star
) -> Atoms:
    """The draws updated by one row: the draws ``finalize`` returns for its ``stats``.

    Args:
        model: the row's observation model.
        payload: the payload of the row.
        atoms: the draws.
        star: a pytree of scalar leaves describing the star.

    Returns:
        The updated draws.
    """
    return model.finalize(model.stats(payload, atoms, star), atoms, star)[1]


CELL_SAMPLES = 4096
"""Samples per draw in ``binned_cells``."""


def binned_cells(
    model: AbstractObservationModel,
    atoms: Atoms,
    action: Action,
    cell: Callable[[PyTree], Int[Array, ""]],
    n_cells: int,
) -> Float[Array, "A Y"]:
    """Outcome-cell probabilities binned from ``CELL_SAMPLES`` draws of ``sample``.

    A Monte Carlo estimate with a fixed key, for a model without a closed form; a
    closed form is tested against it.

    Args:
        model: the observation model.
        atoms: the draws.
        action: the action.
        cell: the index of the outcome cell one payload falls in.
        n_cells: the number of outcome cells, ``Y``, static.

    Returns:
        Cell probabilities of shape ``(A, Y)``; each row sums to one.
    """
    base = jax.random.key(0)
    keys = jax.vmap(jax.random.fold_in, in_axes=(None, 0))(
        base, jnp.arange(CELL_SAMPLES)
    )
    payloads = jax.vmap(model.sample, in_axes=(None, None, 0))(atoms, action, keys)
    index = jax.vmap(jax.vmap(cell))(payloads)  # (n, A)
    counts = jax.nn.one_hot(index, n_cells).sum(axis=0)
    return counts / CELL_SAMPLES
