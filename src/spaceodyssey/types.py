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


def as_float(x) -> Float[Array, "..."]:
    """``x`` as the default float type, never weakly typed, in any shape."""
    return jnp.asarray(x, dtype=float)


def _optional_array(x):
    """``x`` as a JAX array, or ``None`` when not declared."""
    return None if x is None else jnp.asarray(x)


class WeightedCloud(eqx.Module):
    """Weighted draws from one refit; a stand-in with a sample posterior's fields.

    ``samples`` is a named pytree whose leaves share the leading draw axis ``A``, every
    leaf of shape ``(A, *shape)``.

    Weight contract: ``log_weights`` is the log of target over proposal, where the
    target is ``p0`` times the rows the refit represents (``Refit.row_mask``). It is
    not a log ratio against a proposal that equals ``p0``: that reading would count
    ``-log p0`` twice when the role weights are formed. One cloud has one row set, so
    one mask serves all its draws; a block normalization would need one per draw. The
    cloud comes from one refit.

    ``log_ref_density`` is ``log p0`` of each draw, a density on the coordinates named
    by ``chart``. Reweighting the draws to a role prior on the same chart needs both;
    a cloud that declares neither can be held but not reweighted.

    Attributes:
        samples: the state pytree, every leaf of shape ``(A, *shape)``.
        log_weights: log of target over proposal, shape ``(A,)``.
        evidence: log marginal likelihood ``log Z`` of the refit, a scalar; NaN
            where the refit does not estimate it (unknown, not zero).
        log_ref_density: log of ``p0`` on ``chart``, shape ``(A,)``, or ``None`` when
            not declared.
        chart: the coordinates the leaves of ``samples`` are in, a static label, or
            ``None`` when not declared.
        epoch: the zero of every time in the draws, static, or ``None`` when not
            declared.

    Raises:
        ValueError: if ``log_ref_density`` is declared without a ``chart``.
    """

    samples: PyTree = eqx.field(converter=_asarrays)
    log_weights: Float[Array, " A"] = eqx.field(converter=jnp.asarray)
    evidence: Float[Array, ""] = eqx.field(converter=as_float)
    log_ref_density: Float[Array, " A"] | None = eqx.field(
        converter=_optional_array, default=None
    )
    chart: str | None = eqx.field(static=True, default=None, kw_only=True)
    epoch: object | None = eqx.field(static=True, default=None, kw_only=True)

    def __check_init__(self):
        """Check that a declared reference density names its chart."""
        if self.log_ref_density is not None and self.chart is None:
            raise ValueError("a cloud that declares log_ref_density declares its chart")


class Atoms[StateT](eqx.Module):
    """A weighted cloud with its gas block.

    ``cloud.samples`` carries the state of type ``StateT``. The gas block holds, per
    draw, planet and gas, a Gaussian over the abundance: ``x_mean`` and ``x_var`` (zero
    variance is a point state).

    ``K`` is the number of planets one draw can hold; a draw with fewer is padded, and
    ``exists`` marks its real slots. A slot that does not exist is claim-false whatever
    its state, so padding a draw changes no probability. Its state is masked, not
    read, in the values; keep it finite all the same (zeros, say), because a NaN in a
    masked branch still turns a derivative into NaN. A draw padded only to give every
    star one shape carries zero weight (``log_weights = -inf``).

    Attributes:
        cloud: the weighted draws.
        x_mean: mean abundance, shape ``(A, K, G)``.
        x_var: abundance variance, shape ``(A, K, G)``, zero for a point state.
        exists: whether each planet slot of each draw holds a planet, shape ``(A, K)``.
    """

    cloud: WeightedCloud
    x_mean: Float[Array, "A K G"] = eqx.field(converter=jnp.asarray)
    x_var: Float[Array, "A K G"] = eqx.field(converter=jnp.asarray)
    exists: Bool[Array, "A K"] = eqx.field(converter=jnp.asarray)


def as_index(x) -> Int[Array, "..."]:
    """``x`` as the default integer type, never weakly typed, in any shape.

    An index built from a Python integer would otherwise be weakly typed, and the
    strongly typed copy a compiled function returns would trace that function again.
    Every integer field a host loop feeds back into compiled code converts with this,
    and every such float field with ``as_float``.
    """
    return jnp.asarray(x, dtype=int)


class Refit(eqx.Module):
    """The draws one refit returned for one star, and the log rows they represent.

    The cloud's weights already carry the rows in ``row_mask``; ``replay`` applies the
    star's other rows. A belief's draws are ``Atoms``, never a ``Refit``: their gas
    block has been updated by the rows replayed into it, so they cannot be replayed
    again.

    Attributes:
        atoms: the weighted draws, as the refit returned them.
        row_mask: the log rows the refit conditioned on, shape ``(W,)``, one set for
            the whole cloud.
        star_index: the star's index in the log's ``star`` column.
    """

    atoms: Atoms
    row_mask: Bool[Array, " W"] = eqx.field(converter=jnp.asarray)
    star_index: Int[Array, ""] = eqx.field(converter=as_index)


class Pool(eqx.Module):
    """The undetected planets at a star, per role.

    Each mark is one planet (``K = 1`` in ``marks``); the marks carry the conditional
    distribution of an undetected planet and the pool carries how many there are in
    expectation. A mark's weight under role ``r`` is formed like a draw's,
    ``log_weights + log_prior[r] - log_ref_density + log_like[r]``, normalized over the
    marks.

    A null observation thins the pool: it lowers ``log_mass`` and adds the log
    probability of not being detected to ``log_like``. The pool at time zero (after
    the grant, which enters at time zero) is a ``Pool`` of its own, held by the caller
    beside the current one, so the claim-true rate then can be set against the rate
    now.

    Attributes:
        marks: the marks, leading axis ``Am``.
        log_prior: log prior of each mark under each role, shape ``(R, Am)``.
        log_like: log likelihood of each mark under each role, shape ``(R, Am)``.
        log_mass: log expected number of planets, per role, shape ``(R,)``.

    Raises:
        ValueError: if a mark holds more than one planet, or the fields disagree on
            the number of roles or marks.
    """

    marks: Atoms
    log_prior: Float[Array, "R Am"] = eqx.field(converter=jnp.asarray)
    log_like: Float[Array, "R Am"] = eqx.field(converter=jnp.asarray)
    log_mass: Float[Array, " R"] = eqx.field(converter=jnp.asarray)

    def __check_init__(self):
        """Check the static shapes: one planet per mark, one ``R`` and one ``Am``."""
        if self.marks.x_mean.shape[-2] != 1:
            raise ValueError(
                f"a mark is one planet (K = 1), not K = {self.marks.x_mean.shape[-2]}"
            )
        n_mark = self.marks.cloud.log_weights.shape[-1]
        per_mark = (self.log_prior, self.log_like)
        roles = {x.shape[-2] for x in per_mark} | {self.log_mass.shape[-1]}
        if len(roles) != 1 or {x.shape[-1] for x in per_mark} != {n_mark}:
            raise ValueError(
                f"pool fields disagree on R or Am (log_prior {self.log_prior.shape}, "
                f"log_like {self.log_like.shape}, log_mass {self.log_mass.shape}, "
                f"{n_mark} marks)"
            )


ACCEPTED = 0
"""Refusal code of an accepted row."""

OVER_BUDGET = 1
"""Refusal code of a row whose exposure exceeds the budget left: charged nothing, its
payload zeros."""


class Row(eqx.Module):
    """One released observation result.

    The log is a ``Row`` stacked along a leading axis (``ObservationLog``), so its
    columns are these fields. Times are calendar days from the start of the survey;
    the budget counts only ``charged_d``.

    Attributes:
        star: star index.
        mode: observing mode index.
        time_d: start time, calendar days from the start of the survey.
        exposure_d: exposure length in days.
        charged_d: time charged against the budget in days.
        release_d: calendar time at which the row is released, in days.
        refusal: refusal code, ``ACCEPTED`` or ``OVER_BUDGET``.
        model_slot: index into the run's model table of the model that inverts the
            payload.
        payload: one payload per model, each a dict of named arrays, zero-padded; only
            the payload in ``model_slot`` is meaningful.
    """

    star: Int[Array, ""] = eqx.field(converter=as_index)
    mode: Int[Array, ""] = eqx.field(converter=as_index)
    time_d: Float[Array, ""] = eqx.field(converter=as_float)
    exposure_d: Float[Array, ""] = eqx.field(converter=as_float)
    charged_d: Float[Array, ""] = eqx.field(converter=as_float)
    release_d: Float[Array, ""] = eqx.field(converter=as_float)
    refusal: Int[Array, ""] = eqx.field(converter=as_index)
    model_slot: Int[Array, ""] = eqx.field(converter=as_index)
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
    count: Int[Array, ""] = eqx.field(converter=as_index)

    def __check_init__(self):
        """Check the static shapes: one capacity ``W`` across the columns."""
        capacity = {jnp.shape(x)[0] for x in jax.tree.leaves(self.rows)}
        if len(capacity) != 1:
            raise ValueError(f"log columns disagree on the capacity W: {capacity}")


def real_rows(log: ObservationLog) -> Bool[Array, " W"]:
    """Which rows of a log are real: those before ``count``, the rest padding.

    Args:
        log: the log.

    Returns:
        True for each real row, shape ``(W,)``.
    """
    return jnp.arange(log.rows.star.shape[0]) < log.count


def spent_d(log: ObservationLog) -> Float[Array, ""]:
    """The time a run has charged against its budget so far, days.

    The grant and a refused row are charged nothing; waiting is charged nothing.

    Args:
        log: the log.

    Returns:
        The charged time of the rows before ``count``.
    """
    return jnp.sum(jnp.where(real_rows(log), log.rows.charged_d, 0.0))


def clock_d(log: ObservationLog) -> Float[Array, ""]:
    """The calendar time at which the instrument is next free, days: the survey clock.

    Observations run one after another on the calendar, so the clock is the latest end
    of a real row, ``time_d + charged_d``; zero for a log with no rows. Calendar time
    advances by exposures and by waiting: a row that starts later than the one before
    it ended moves the clock by the wait as well, and a refused row by its wait alone.

    Args:
        log: the log.

    Returns:
        The calendar time the rows before ``count`` end.
    """
    rows = log.rows
    ends = jnp.where(real_rows(log), rows.time_d + rows.charged_d, 0.0)
    return jnp.max(ends, initial=0.0)


class Action(eqx.Module):
    """What a policy asks for: one observation of one star in one mode.

    An action starts at the later of ``not_before_d`` and the time the instrument is
    next free (``clock_d``), so a policy waits by asking for a later start; waiting
    costs calendar time and no budget. A granted action's ``not_before_d`` is ignored:
    the grant is released at time zero.

    Attributes:
        star: star index.
        mode: index of the observing mode in the design's mode table.
        exposure_d: requested exposure in days.
        not_before_d: earliest calendar start, days from the start of the survey;
            zero (as soon as the instrument is free) by default, and broadcast to the
            shape of ``exposure_d``.
    """

    star: Int[Array, ""] = eqx.field(converter=as_index)
    mode: Int[Array, ""] = eqx.field(converter=as_index)
    exposure_d: Float[Array, ""] = eqx.field(converter=as_float)
    not_before_d: Float[Array, ""] = eqx.field(converter=as_float)

    def __init__(self, star, mode, exposure_d, not_before_d=0.0):
        """An action; ``not_before_d`` is broadcast to the shape of ``exposure_d``."""
        self.star = star
        self.mode = mode
        self.exposure_d = as_float(exposure_d)
        self.not_before_d = jnp.broadcast_to(
            as_float(not_before_d), self.exposure_d.shape
        )


class Grant(eqx.Module):
    """The actions a grant reveals at time zero, stacked along a leading axis.

    A grant holds no data: the world draws each granted row from its truth, as for an
    observation, and releases it at time zero, charged nothing. Granted row ``i`` is
    row ``i`` of the log. Each row's model is read from the design's mode table
    through its action's mode.

    Attributes:
        actions: an ``Action`` whose leaves have a leading axis over the granted rows;
            each gives the star, the mode and the exposure that sets the S/N.
    """

    actions: Action


class AbstractObservationModel(eqx.Module):
    """The contract of an observation model, a row's payload given a draw and an action.

    A model holds configuration only, never an observed datum; it is handed a payload
    each time it scores one. Its outcomes are also offered as discrete cells
    (``cells``), which a planner reads ahead of the data. That is a deliberate
    difference from a data-bearing likelihood term: the log is a fixed-capacity buffer
    of payloads per model, scored by one commutative sum per model.

    A planner reads cells on point draws only (``x_var = 0``; ``quadrature_draws``
    splits an uncertain gas block into point sub-draws first). Given a point draw the
    cells of every row are independent, so the probability of a cell under each draw
    is the exact update of the weights by that outcome and no payload is needed.

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

    In a run's model table, every model's statistics are taken against the draws as
    the refit returned them, while ``finalize`` runs slot by slot in table order, each
    against the draws the slot before it returned. A model's ``stats`` must therefore
    read nothing of a draw that a model in an earlier slot changes in ``finalize``. The
    toy models qualify: the Gaussian model's statistics read no draw, and the step
    report's read only point gas blocks, which no ``finalize`` changes.

    Each model declares ``searches``, a class constant: whether its rows detect
    planets. Only a searching model takes planets out of the pool or has a
    completeness; a characterization (a spectrum of known planets) leaves the pool as
    it is.

    Attributes:
        searches: whether the model detects planets, a class constant every concrete
            model sets.
    """

    searches: eqx.AbstractClassVar[bool]

    @abc.abstractmethod
    def sample(self, key: PRNGKeyArray, atoms: Atoms, action: Action) -> PyTree:
        """Draw the payload an action would return, independently for each draw.

        Args:
            key: the key of the row.
            atoms: the draws.
            action: the action.

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

        For a searching model (``searches``) cell 0 is the miss: on a pool's marks,
        every other cell is a detection. A model without a closed form returns
        ``binned_cells``.

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
    payloads = jax.vmap(model.sample, in_axes=(0, None, None))(keys, atoms, action)
    index = jax.vmap(jax.vmap(cell))(payloads)  # (n, A)
    counts = jax.nn.one_hot(index, n_cells).sum(axis=0)
    return counts / CELL_SAMPLES
