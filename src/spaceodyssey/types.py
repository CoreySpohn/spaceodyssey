"""Leaf types shared by every layer: draws, pool, rows, actions, model and refit.

This module imports nothing from the package. Axis names used in the annotations:
``A`` draws, ``K`` planets one draw can hold (a star's planet capacity), ``G``
gases, ``R`` roles (forecast is row 0, the verdict members are rows 1 to ``R - 1``),
``W`` log row capacity and ``Am`` the atoms of a pool.
"""

import abc
from collections.abc import Callable
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Float64, Int, PRNGKeyArray, PyTree


def _asarrays(tree):
    """Convert every leaf of ``tree`` to a JAX array."""
    return jax.tree.map(jnp.asarray, tree)


def as_float(x) -> Float[Array, "..."]:
    """``x`` as the default float type, never weakly typed, in any shape."""
    return jnp.asarray(x, dtype=float)


def _as_float64(x) -> Float64[Array, "..."]:
    """``x`` as a float64 array (needs ``jax_enable_x64``)."""
    return jnp.asarray(x, dtype=jnp.float64)


class AbstractRolePrior(eqx.Module):
    """A role prior: a joint density over whole draws on a declared chart.

    The density is taken in the coordinates named by ``chart``. Two priors whose ratio
    is formed agree on a chart, or the change of variables (the log Jacobian) is
    missing from one side and the ratio is tilted. The density covers a whole draw
    (its planets, their states and gas blocks, and which slots exist) and need not
    factor over them, which a law over one quantity does not express.

    Attributes:
        chart: the coordinates in which the density is taken, a static label.
    """

    chart: str = eqx.field(static=True)

    @abc.abstractmethod
    def log_prob(self, atoms: "Atoms", star) -> Float[Array, " A"]:
        """Log prior density of each draw on ``chart``.

        A draw outside the prior's support returns ``-inf`` (through ``jnp.where``, so
        the gradient stays finite), never ``log(0)``.

        Args:
            atoms: the weighted draws.
            star: a pytree of scalar leaves describing the star.

        Returns:
            Log densities of shape ``(A,)``.
        """

    @abc.abstractmethod
    def sample(self, key: PRNGKeyArray, star) -> "Atoms":
        """Draw one point state of a star from the prior.

        Args:
            key: the key of the draw.
            star: a pytree of scalar leaves describing the star.

        Returns:
            Atoms holding one draw (``A = 1``) with a point gas block (``x_var = 0``),
            on this prior's ``chart``.
        """


class WeightedCloud(eqx.Module):
    """Weighted draws from one refit; a stand-in with a posterior's fields.

    ``samples`` is a named pytree whose leaves share the leading draw axis ``A``, every
    leaf of shape ``(A, *shape)``.

    Weight contract: ``log_weights`` is the log of target over proposal, where the
    target is the refit's prior ``prior`` times the rows the refit represents
    (``Refit.row_mask``). One cloud has one row set, so one mask serves all its draws.
    The cloud comes from one refit.

    ``prior`` is that refit's prior ``pi_0``, a role-prior record; it stands where a
    posterior's fit carries its priors. Reweighting the draws to a role prior ``pi_r``
    multiplies the weights by ``pi_r / pi_0``, both evaluated on the draws, so the two
    priors share a chart. ``log_target`` is the log density of the refit's target at
    each draw, prior times likelihood, normalized, on the coordinates of the draws; no
    weight is formed from it.

    Attributes:
        samples: the state pytree, every leaf of shape ``(A, *shape)``.
        log_weights: log of target over proposal, shape ``(A,)``.
        log_target: log of the refit's normalized target at each draw, float64,
            shape ``(A,)``.
        log_evidence: log marginal likelihood ``log Z`` of the refit, a scalar; NaN
            where the refit does not estimate it (unknown, not zero).
        prior: the refit's prior ``pi_0``.

    Raises:
        ValueError: if ``log_target`` is not float64 (``jax_enable_x64`` is off) or
            its shape differs from that of ``log_weights``.
    """

    samples: PyTree = eqx.field(converter=_asarrays)
    log_weights: Float[Array, " A"] = eqx.field(converter=jnp.asarray)
    log_target: Float64[Array, " A"] = eqx.field(converter=_as_float64)
    log_evidence: Float[Array, ""] = eqx.field(converter=as_float)
    prior: AbstractRolePrior

    def __check_init__(self):
        """Check that ``log_target`` is float64 and has one entry per draw."""
        if self.log_target.dtype != jnp.float64:
            raise ValueError("log_target is float64: enable jax_enable_x64")
        if self.log_target.shape != self.log_weights.shape:
            raise ValueError(
                f"log_target {self.log_target.shape} and log_weights "
                f"{self.log_weights.shape} disagree on the draws"
            )


class GasNodes(eqx.Module):
    """Quadrature nodes of a gas block: each factor a weighted set of point abundances.

    A factor is one planet and one gas of one draw. Given the draw the factors are
    independent, so each carries its own nodes, and the block is their product without
    ever forming it: ``n`` nodes per factor stand for ``n ** (K G)`` points of the
    block. A planner's hypothetical outcome reweights the nodes of each factor it reads
    (``update_by_cell``).

    The weights are held linear, not as logs: a planner reweights and sums them at
    every node of its lookahead, and the sums need no exponential.

    Attributes:
        x: the abundance at each node, shape ``(A, K, G, n)``.
        weights: the weight of each node, normalized per factor (each factor's weights
            sum to one), shape ``(A, K, G, n)``.
    """

    x: Float[Array, "A K G n"] = eqx.field(converter=jnp.asarray)
    weights: Float[Array, "A K G n"] = eqx.field(converter=jnp.asarray)


class Atoms[StateT](eqx.Module):
    """A weighted cloud with its gas block.

    ``cloud.samples`` carries the state of type ``StateT``. The gas block holds, per
    draw, planet and gas, a Gaussian over the abundance: ``x_mean`` and ``x_var`` (zero
    variance is a point state). A planner's draws may carry the block on quadrature
    nodes instead (``nodes``, set by ``quadrature_draws``); every reader of the gas
    block then reads the nodes (``GasAbove`` does; a model's cells are read at each
    node through ``gas_cells``), and ``x_mean`` and ``x_var`` keep the block the nodes
    were built from: stale once ``nodes`` is set, never to be read as the belief's
    block (``outcome_cells`` hands a model without ``gas_cells`` a NaN block there).

    ``K`` is the number of planets one draw can hold; a draw with fewer is padded, and
    ``exists`` marks its real slots. A slot that does not exist is claim-false whatever
    its state, so padding a draw changes no probability of a claim. The claim and the
    values mask it; a model's likelihood may read it (the toy's gas models read a
    padded slot's gas, so a gas reading is an existence-free likelihood only when the
    tracked slot exists on every draw). Keep it finite all the same (zeros, say),
    because a NaN in a masked branch still turns a derivative into NaN. A draw padded
    only to give every star one shape carries zero weight (``log_weights = -inf``).

    Attributes:
        cloud: the weighted draws.
        x_mean: mean abundance, shape ``(A, K, G)``.
        x_var: abundance variance, shape ``(A, K, G)``, zero for a point state.
        exists: whether each planet slot of each draw holds a planet, shape ``(A, K)``.
        nodes: the gas block on quadrature nodes, or None (the block is ``x_mean`` and
            ``x_var``).
    """

    cloud: WeightedCloud
    x_mean: Float[Array, "A K G"] = eqx.field(converter=jnp.asarray)
    x_var: Float[Array, "A K G"] = eqx.field(converter=jnp.asarray)
    exists: Bool[Array, "A K"] = eqx.field(converter=jnp.asarray)
    nodes: GasNodes | None = None


def planet_tracks(atoms: Atoms) -> Bool[Array, " K"]:
    """The planet slots a cloud holds a track in: those some draw holds a planet in.

    ``K`` is the planet capacity of a star, static and padded: a slot no draw holds is
    free, and a detection there is of a planet new to the cloud (association by truth
    label, declared deviation 2: the payload's slot is the truth's). A new planet is
    not scored against the draws (it cannot be, none holds it); it is a pool event,
    already counted found by the pool's replay, and a refit's job
    (``AbstractRefitter``).

    Args:
        atoms: the draws of one star.

    Returns:
        True for each slot some draw holds a planet in, shape ``(K,)``.
    """
    return jnp.any(atoms.exists, axis=0)


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


POISSON = "poisson"
"""A pool law: the number of undetected planets is Poisson, mean ``exp(log_mass)``."""

BERNOULLI = "bernoulli"
"""A pool law: at most one undetected planet, present with probability
``exp(log_mass)``."""


class Pool(eqx.Module):
    """The undetected planets at a star, per role.

    Each mark is one planet (``K = 1`` in ``marks``); the marks carry the conditional
    distribution of an undetected planet and the pool carries how many there are in
    expectation. A mark's weight under role ``r`` is formed like a draw's,
    ``log_weights + log_prior_ratio[r] + log_like[r]``, normalized over the marks, with
    ``log_prior_ratio`` the role's prior over that of the marks' cloud.

    The law of the number of planets is static (``law``): ``POISSON``, a Poisson count
    of mean ``exp(log_mass)``, or ``BERNOULLI``, at most one planet, present with
    probability ``exp(log_mass)`` (so ``log_mass <= 0``). The expected number is
    ``exp(log_mass)`` either way; the laws differ in what a search's outcome says
    about the planets left. A Poisson pool thinned by a search is the planets it
    missed, Poisson with the thinned rate whatever it found, so its update reads no
    payload; a Bernoulli pool whose planet is found is empty, and one that a search
    missed holds its planet with the posterior probability
    ``rho (1 - dbar) / (1 - rho dbar)`` (``value.thin``, ``value.replay_pool``).

    A null observation thins the pool: it lowers ``log_mass`` and adds the log
    probability of not being detected to ``log_like``. The pool at time zero (after
    the grant, which enters at time zero) is a ``Pool`` of its own, held by the caller
    beside the current one, so the claim-true rate then can be set against the rate
    now.

    Attributes:
        marks: the marks, leading axis ``Am``.
        log_prior_ratio: log of each role's prior over the marks' cloud prior, per
            mark, shape ``(R, Am)``.
        log_like: log likelihood of each mark under each role, shape ``(R, Am)``.
        log_mass: log expected number of planets, per role, shape ``(R,)``.
        law: the law of the number of planets, ``POISSON`` (the default) or
            ``BERNOULLI``, static.

    Raises:
        ValueError: if a mark holds more than one planet, the fields disagree on the
            number of roles or marks, or the law is neither.
    """

    marks: Atoms
    log_prior_ratio: Float[Array, "R Am"] = eqx.field(converter=jnp.asarray)
    log_like: Float[Array, "R Am"] = eqx.field(converter=jnp.asarray)
    log_mass: Float[Array, " R"] = eqx.field(converter=jnp.asarray)
    law: str = eqx.field(static=True, default=POISSON)

    def __check_init__(self):
        """Check the law and the shapes: one planet per mark, one ``R`` and ``Am``."""
        if self.law not in (POISSON, BERNOULLI):
            raise ValueError(f"the pool law is {POISSON!r} or {BERNOULLI!r}")
        if self.marks.x_mean.shape[-2] != 1:
            raise ValueError(
                f"a mark is one planet (K = 1), not K = {self.marks.x_mean.shape[-2]}"
            )
        n_mark = self.marks.cloud.log_weights.shape[-1]
        per_mark = (self.log_prior_ratio, self.log_like)
        roles = {x.shape[-2] for x in per_mark} | {self.log_mass.shape[-1]}
        if len(roles) != 1 or {x.shape[-1] for x in per_mark} != {n_mark}:
            raise ValueError(
                "pool fields disagree on R or Am (log_prior_ratio "
                f"{self.log_prior_ratio.shape}, "
                f"log_like {self.log_like.shape}, log_mass {self.log_mass.shape}, "
                f"{n_mark} marks)"
            )


ACCEPTED = 0
"""Refusal code of an accepted row."""

OVER_BUDGET = 1
"""Refusal code of a row whose exposure exceeds the budget left: charged nothing, its
payload zeros."""

WAITED = 2
"""Code of a wait, a row of zero exposure: the instrument idles until the row's start.
Charged nothing, its payload zeros, released at its start; it carries no data, so no
replay applies it and ``score`` stages it as no observation, but it moves the clock."""

PAST_END = 3
"""Refusal code of a row that would start after the calendar's end
(``Resources.end_d``), or whose star never leaves keepout before it: charged nothing,
its payload zeros."""


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
        refusal: refusal code, ``ACCEPTED``, ``OVER_BUDGET``, ``WAITED`` or
            ``PAST_END``.
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

    ``time_d`` is the action's start: asked for, the earliest start; realized, the
    start itself, as a row's ``time_d`` is. An action starts at the later of
    ``time_d`` and the time the instrument is next free (``clock_d``), and later still
    while its star is in keepout, so a policy waits by asking for a later start;
    waiting costs calendar time and no budget. A granted action's ``time_d`` is
    ignored: the grant is released at time zero.

    A model reads an action's start as its ``time_d``: ``step`` hands the world the
    action with ``time_d`` set to the realized start, a replay of a real row rebuilds
    it from the row's ``time_d``, and a planner's lookahead gives its actions the
    planned start. A model whose rows depend on the start (a search whose
    detection probability follows the orbital phase or the zodi) records it in its
    payload.

    Attributes:
        star: star index.
        mode: index of the observing mode in the design's mode table.
        exposure_d: requested exposure in days.
        time_d: the start, days from the start of the survey: asked for, the
            earliest start, zero (as soon as the instrument is free) by default;
            realized, the start itself. Broadcast to the shape of ``exposure_d``.
    """

    star: Int[Array, ""] = eqx.field(converter=as_index)
    mode: Int[Array, ""] = eqx.field(converter=as_index)
    exposure_d: Float[Array, ""] = eqx.field(converter=as_float)
    time_d: Float[Array, ""] = eqx.field(converter=as_float)

    def __init__(self, star, mode, exposure_d, time_d=0.0):
        """An action; ``time_d`` is broadcast to the shape of ``exposure_d``."""
        self.star = star
        self.mode = mode
        self.exposure_d = as_float(exposure_d)
        self.time_d = jnp.broadcast_to(as_float(time_d), self.exposure_d.shape)


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
    (``cells``), which a planner reads ahead of the data, and a real row's payload
    falls in one of them (``cell``), so real outcomes advance a plan as hypothetical
    ones do. That is a deliberate
    difference from a data-bearing likelihood term: the log is a fixed-capacity buffer
    of payloads per model, scored by one commutative sum per model.

    A planner reads cells on point states only: a draw's gas block is a point
    (``x_var = 0``) or carried on quadrature nodes (``quadrature_draws``), each node a
    point. Given a point the cells of every row are independent, so the probability of
    a cell is the exact update by that outcome and no payload is needed. A model whose
    cells read the gas block says so through ``gas_cells``: each factor's (one planet,
    one gas) probability of each of its bins, read at each node, with ``cells`` their
    joint; the planner then updates each factor's nodes by its own bin, which keeps the
    cost linear in the nodes per factor. A model whose cells read no gas (a search)
    returns None there and is read through ``cells`` alone.

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

    Each model also declares ``stage``, a class constant: the stage of a star's
    follow-up chain that a positive outcome (any cell but cell 0) reaches. A planner
    offers an option once the star has reached the option's required stage (a
    spectrum needs a detection), so stages are ordered and a star's stage only rises.
    A model whose outcomes open nothing reaches stage 0.

    A model whose cells step in the abundance (a one-bit report of a gas above its
    floor) declares the step, ``gas_step``, so the quadrature splits the block there
    (``quadrature_draws``); a model whose cells are smooth in it leaves it None.

    A searching model also says which planet slots a real row detected
    (``detections``), and its ``stats`` score only the detections of tracks
    (``planet_tracks``): a detection in a slot no draw holds is a planet new to the
    cloud, which no draw can explain, so it is not scored (the pool's replay already
    counts it found, and a refit adds its track).

    Attributes:
        searches: whether the model detects planets, a class constant every concrete
            model sets.
        stage: the stage a positive outcome reaches, a class constant every concrete
            model sets.
        gas_step: the abundance at which the cells step, a class constant; None (the
            default) for cells smooth in the abundance or that do not read it.
    """

    searches: eqx.AbstractClassVar[bool]
    stage: eqx.AbstractClassVar[int]
    gas_step: ClassVar[float | None] = None

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
            A pytree of arrays; the statistics of several rows are their leafwise
            sum. A statistic that does not depend on the draw is held without the
            draw axis and broadcast in ``finalize``: a replay sums every row slot of
            the log, filled or not, so a row costs the size of its statistics.
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
    def cell(self, payload: PyTree) -> Int[Array, ""]:
        """The outcome cell of one real row: the cell of ``cells`` its payload is in.

        A planner reads the cells ahead of the data; a run's rows, once released, are
        read back into the same cells, so the stage a star has reached and the misses
        of a search are read off real rows exactly as the lookahead coded them. A zero
        payload (a refused row or a wait) is in cell 0.

        Args:
            payload: the payload of one row, a dict of named arrays.

        Returns:
            The index of the cell, in ``0 .. Y - 1``.
        """

    @abc.abstractmethod
    def cells(self, atoms: Atoms, action: Action) -> Float[Array, "A Y"]:
        """Probability of each outcome cell under each draw.

        For a searching model (``searches``) cell 0 is the miss: on a pool's marks,
        every other cell is a detection. The cells agree with the likelihood
        (``stats``, ``finalize``): reweighting the draws by a cell's probability is
        the update a row in that cell makes, so a model whose likelihood does not
        score whether a planet exists does not learn it from its cells either. Such a
        model reads a slot some draw holds (``planet_tracks``) on every draw, the
        draws without the planet included; a slot no draw holds has one outcome, as
        an unread one does (code 0 of its factors), so the cells it cannot give have
        probability zero and a planner may drop them (``frontier``'s ``max_cells``).
        A model without a closed form returns ``binned_cells``.

        Args:
            atoms: the draws.
            action: the action.

        Returns:
            Cell probabilities of shape ``(A, Y)``; each row sums to one.
        """

    def detections(self, payload: PyTree) -> Bool[Array, " K"] | None:
        """The planet slots a real row of a searching model detected.

        Slots follow the truth's labels (declared deviation 2); a zero payload (a
        refused row or a wait) detects nothing. A model that does not search returns
        None, the default; every searching model overrides it.

        Args:
            payload: the payload of one row, a dict of named arrays.

        Returns:
            True for each slot detected, shape ``(K,)``; or None.
        """
        return None

    def gas_cells(self, atoms: Atoms, action: Action) -> Float[Array, "A K G B"] | None:
        """Each gas factor's probability of each of its bins, on point gas blocks.

        A model whose cells read the gas block returns them, and its ``cells`` is
        their joint: the probability of cell ``y`` is the product over the factors
        (planets, then gases, flattened) of each factor's probability of its bin, the
        cell index mixed radix in the bins, the first factor most significant. The
        default, None, is a model whose cells read no gas; on a belief on quadrature
        nodes its ``cells`` are read with a NaN gas block, and a model that reads the
        gas anyway is refused (``outcome_cells``).

        Args:
            atoms: the draws, point gas blocks.
            action: the action.

        Returns:
            Bin probabilities of shape ``(A, K, G, B)``, each factor's summing to one;
            or None.
        """
        return None


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


class RefitTerm(eqx.Module):
    """The rows a refit adds to a star's cloud: the new likelihood term.

    Attributes:
        log: the run's log, all stars: the public record a refit reads.
        rows: the rows the term adds, shape ``(W,)``: the star's accepted rows
            released by the query time that the cloud does not yet represent
            (``due_rows``).
        models: the run's model table, static in structure; ``Row.model_slot``
            indexes it.
        star: a pytree of scalar leaves describing the star.
    """

    log: ObservationLog
    rows: Bool[Array, " W"]
    models: tuple[AbstractObservationModel, ...]
    star: PyTree


class AbstractRefitter(eqx.Module):
    """The producer of a star's refits: a new cloud from its cloud and its new rows.

    A planner appends a star's released rows to its replay first (``absorb``) and gates
    the weights they leave (``refused_after_rows``); it queues a refit of the star when
    those rows detected a planet its cloud holds no track for (``planet_tracks``) or
    left a role's weights below the gate. The refit returns a cloud that represents the
    old cloud's rows and the term's, any new planet a track in a free slot of the
    star's planet capacity ``K``. It reads only the public log. A refitter is a record
    of configuration (the backend); the toy's is an importance refit
    (``toy.ToyRefitter``), and a study of real data plugs its fitter in here.

    The call has the shape of a posterior update by one likelihood term: the star's
    current cloud, the new rows as the term, the key, and the reweighting gate's
    thresholds (the planner's ``PlannerSettings``), so a fitter whose update takes
    ``(posterior, term, backend, key, *, min_n_eff, max_khat)`` is wrapped directly,
    the term swapped in for the rows the cloud represents. ``proposal_factor`` scales
    the backend's draw count (the proposals it weighs to make the returned draws), so
    a refused refit can be retried with more. A cloud the gate refuses carries NaN log
    weights, never a cloud planned on silently. The thresholds are checked on the host
    where they are set (``PlannerSettings`` refuses NaN). A decision's queue is refit
    in one call, ``refit_stars``.

    The new refit's pool is not the refitter's job: the planner thins the old pool by
    the searches the new refit represents (``replay_pool``), which counts the new
    planet found whatever the refit makes of it.

    A frozen forecast (``PlannerSettings.forecast_updates`` false) is the time-zero
    replay of the planets it knew; when a refit adds a planet, the forecast is
    extended by that planet at its time-zero belief, the refit prior's law of a planet
    in that slot, with no row (``add_tracks``), so the frozen forecast predicts a
    found planet's rows from its prior.
    """

    @abc.abstractmethod
    def refit(
        self,
        refit: Refit,
        term: RefitTerm,
        key: PRNGKeyArray,
        *,
        min_n_eff: Float[Array, ""],
        max_khat: Float[Array, ""],
        proposal_factor: int = 1,
    ) -> Refit:
        """A new refit of one star: its cloud updated by the term's rows.

        Args:
            refit: the star's refit now.
            term: the rows to add, of the same star.
            key: the key of the refit.
            min_n_eff: the least Kish size of the new cloud's weights.
            max_khat: the largest Pareto shape of the new cloud's weights.
            proposal_factor: how many times its backend's proposals the refit draws,
                static; 1 by default.

        Returns:
            The new refit: its draws, of the same ``A`` and ``K``, NaN weights where
            the gate refuses them; its ``row_mask``, the old one and the term's rows;
            and the star's index.
        """

    @abc.abstractmethod
    def add_tracks(
        self, refit: Refit, slots: Bool[Array, " K"], key: PRNGKeyArray, star
    ) -> Refit:
        """The refit with a track in each slot of ``slots``, drawn at its prior.

        Each draw gets a planet in each slot, its state drawn from the refit prior's
        law of a planet there, with no row: the planet's belief at time zero. The
        weights, the other slots and ``row_mask`` are kept.

        Args:
            refit: the star's refit.
            slots: the slots to add a track in, shape ``(K,)``.
            key: the key of the draws.
            star: a pytree of scalar leaves describing the star.

        Returns:
            The refit with the tracks added.
        """
