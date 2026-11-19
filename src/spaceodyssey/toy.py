"""The toy: its state, its observation models, its prior and a survey built from them.

The toy state holds, per draw and planet slot, two bits (``"hz"``, the planet is in the
habitable zone; ``"size"``, it is Earth-sized) and a gas block; ``InHz`` and ``InSize``
read the bits, ``ToyPrior`` is a role prior over whole draws, ``ToyRefitter`` turns a
planet a search finds into a track, and ``toy_survey`` builds a study of many stars on
it, with its menu, grant, keepout table and refitter.

An observation model (``AbstractObservationModel``, defined in ``types``) turns an
action on a star into a row payload (``sample``), scores rows against every draw through
additive statistics (``stats``, ``finalize``), and gives the distribution of a row's
outcome over discrete cells (``cells``). The host passes the model; nothing dispatches
on a traced model id.

The two gas models read the gas block of ``Atoms`` and write one payload per row, a
dict of named ``(K, G)`` arrays: ``"snr"``, the S/N reached by every planet and gas
(zero for a gas the mode does not cover), and the reading (``"reading"`` for the
Gaussian model, ``"bit"`` for the step report). The search (``DetectionModel``) reads
the planets' slots and writes whether each was detected.

Toy constants. The floor, the width of the verdict's gas prior and the linear shape of
the reading are proposed for the toy and wait on approval:

- ``GAS_FLOOR``: 0 dex. Abundance is measured as height above the floor.
- ``GAS_PRIOR_SD``: 1 dex. The verdict's gas prior is ``N(GAS_FLOOR, GAS_PRIOR_SD**2)``,
  so a gas starts at even odds of being above its floor.
- The reading is linear at every S/N; there is no separate shape below detection.
- These models are never used for a negative certificate.
"""

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.special import gammainc, gammaln, logsumexp, ndtr
from jax.scipy.stats import norm
from jaxtyping import Array, Float, Int

from spaceodyssey._numerics import guarded_log, safe_ratio
from spaceodyssey.belief import (
    detected_slots,
    gate_weights,
    replay,
    replay_likelihood,
    role_pool,
)
from spaceodyssey.claim import AbstractConjunct, Claim, GasAbove
from spaceodyssey.design import Design, Keepout, Mode, Resources
from spaceodyssey.roles import Roles
from spaceodyssey.study import Study
from spaceodyssey.types import (
    BERNOULLI,
    POISSON,
    AbstractObservationModel,
    AbstractRefitter,
    AbstractRolePrior,
    Action,
    Atoms,
    Grant,
    Refit,
    WeightedCloud,
    planet_tracks,
)
from spaceodyssey.value import Menu

GAS_FLOOR = 0.0
"""Abundance floor of every gas in the toy, dex (a toy constant)."""

GAS_PRIOR_SD = 1.0
"""Width of the verdict's gas prior at the floor, dex (a toy constant the toy studies
build their priors from; no model reads it)."""

READING_SD_AT_UNIT_SNR = 2.5
"""Noise of a gas reading at S/N 1, dex: ``sigma = 2.5 / (S/N)``."""

STEP_GATE_SNR = 5.0
"""S/N at or above which the step report gives its bit."""


def _snr(snr_one_day: Float[Array, " G"], action: Action) -> Float[Array, " G"]:
    """S/N per gas reached by an action: it grows as the root of the exposure."""
    return snr_one_day * jnp.sqrt(action.exposure_d)


def _read_snr(
    snr_one_day: Float[Array, " G"], atoms: Atoms, action: Action
) -> Float[Array, "A K G"]:
    """S/N of each gas of each slot of each draw; zero for a slot with no planet."""
    snr = jnp.broadcast_to(_snr(snr_one_day, action), atoms.x_mean.shape)
    return jnp.where(atoms.exists[..., None], snr, 0.0)


def _joint(factors: Float[Array, "A K G B"]) -> Float[Array, "A Y"]:
    """Joint cell probabilities of independent factors, ``Y = B ** (K G)``.

    The cell index is mixed radix over the flattened ``(K, G)`` axes, the first factor
    most significant, matching ``_cell_index``.
    """
    n_draw, n_bin = factors.shape[0], factors.shape[-1]
    flat = factors.reshape(n_draw, -1, n_bin)
    joint = flat[:, 0]
    for j in range(1, flat.shape[1]):
        joint = (joint[:, :, None] * flat[:, j, None, :]).reshape(n_draw, -1)
    return joint


def _cell_index(codes: Int[Array, "K G"], n_bin: int) -> Int[Array, ""]:
    """Mixed-radix index of per-factor codes, the first factor most significant."""
    flat = codes.reshape(-1)
    powers = n_bin ** jnp.arange(flat.shape[0] - 1, -1, -1)
    return jnp.sum(flat * powers)


class GaussianGasModel(AbstractObservationModel):
    """A linear Gaussian reading of every gas abundance.

    The reading is ``y = x + eps`` with ``eps ~ N(0, sigma**2)`` and
    ``sigma = READING_SD_AT_UNIT_SNR / (S/N)`` dex: no gravity slope, no bias, and the
    same linear shape at every S/N. Noise is independent across planets and gases. A gas
    with S/N 0 is not covered: its reading is 0 and carries no information.

    The outcome cells bin each reading at ``edges``; an uncovered gas has one outcome
    (cell 0), and so has every gas of a slot with no planet, which is not read (S/N
    0). A row's cells are the joint over planets and gases; ``cell`` and
    ``n_cells`` give the same coding for ``binned_cells``. A planner reads the cells on
    point draws (``quadrature_draws``), where the bins of two rows are independent
    given the draw and reweighting by a bin's probability is the exact update; on an
    uncertain block one bin does not stand for any single reading.

    Two rows reading one gas of a draw with ``x_var > 0`` share its abundance, so
    their per-row likelihoods do not add; ``stats`` and ``finalize`` carry the exact
    conjugate update for any number of rows.

    The model characterizes the planets it reads; it does not search (``searches`` is
    false), so it takes no planet out of the pool.

    Attributes:
        snr_one_day: S/N of each gas after one day of exposure, shape ``(G,)``.
        edges: bin edges of a reading, dex, increasing; default one edge at the floor.
    """

    searches: ClassVar[bool] = False
    stage: ClassVar[int] = 0

    snr_one_day: Float[Array, " G"] = eqx.field(converter=jnp.asarray)
    edges: Float[Array, " E"] = eqx.field(converter=jnp.asarray, default=(GAS_FLOOR,))

    def _noise_var(self, snr: Float[Array, "*s"]) -> Float[Array, "*s"]:
        """Reading variance; zero where the gas is not covered (no reading)."""
        return safe_ratio(READING_SD_AT_UNIT_SNR, snr) ** 2

    def _precision(self, snr: Float[Array, "*s"]) -> Float[Array, "*s"]:
        """Reading precision ``1 / sigma**2``; zero where the gas is not covered."""
        return jnp.where(snr > 0, (snr / READING_SD_AT_UNIT_SNR) ** 2, 0.0)

    def stats(self, payload, atoms, star):
        """Precision-weighted sums of the readings of one row, per planet and gas.

        With precision ``p = 1 / sigma**2`` (zero for an uncovered gas) and reading
        ``y``: ``"precision"`` is ``p``, ``"weighted_sum"`` ``p y``,
        ``"weighted_square_sum"`` ``p y**2`` and ``"log_norm"`` the normalizer
        ``log(p / (2 pi)) / 2`` (zero for an uncovered gas). They do not depend on the
        draw, so they are held per planet and gas, ``(K, G)``, and a row costs the same
        at any number of draws; ``finalize`` broadcasts them over the draws.
        """
        snr, y = payload["snr"], payload["reading"]  # (K, G)
        precision = self._precision(snr)
        safe = jnp.where(snr > 0, precision, 1.0)
        log_norm = jnp.where(snr > 0, 0.5 * jnp.log(safe / (2 * jnp.pi)), 0.0)
        per_gas = {
            "precision": precision,
            "weighted_sum": precision * y,
            "weighted_square_sum": precision * y**2,
            "log_norm": log_norm,
        }
        block = atoms.x_mean.shape[-2:]
        return jax.tree.map(lambda x: jnp.broadcast_to(x, block), per_gas)

    def finalize(self, stats, atoms, star):
        """Exact conjugate update of every draw's gas Gaussian by all summed rows.

        With prior ``N(m, v)`` per gas and total precision ``P``, precision-weighted
        mean ``ybar`` and scatter ``S = sum p y**2 - P ybar**2``, the log marginal of
        the rows is ``log_norm - (S + P (ybar - m)**2 / (1 + v P)) / 2 -
        log(1 + v P) / 2``, summed over planets and gases, and the posterior is
        ``N(m + v P (ybar - m) / (1 + v P), v / (1 + v P))``. A point state (``v = 0``)
        and an uncovered gas (``P = 0``) are unchanged. For one row this is
        ``N(y | m, v + sigma**2)`` and the gain ``v / (v + sigma**2)``.
        """
        m, v = atoms.x_mean, atoms.x_var
        total = stats["precision"]
        ybar = safe_ratio(stats["weighted_sum"], total)
        scatter = stats["weighted_square_sum"] - stats["weighted_sum"] * ybar
        shrink = 1.0 / (1.0 + v * total)
        residual = ybar - m
        terms = stats["log_norm"] - 0.5 * (
            scatter + total * residual**2 * shrink - jnp.log(shrink)
        )
        mean = m + v * total * residual * shrink
        updated = eqx.tree_at(lambda a: (a.x_mean, a.x_var), atoms, (mean, v * shrink))
        return terms.sum(axis=(-2, -1)), updated

    def sample(self, key, atoms, action):
        """Readings of every planet and gas, per draw; ``(A, K, G)`` arrays."""
        snr = _read_snr(self.snr_one_day, atoms, action)
        sd = jnp.sqrt(atoms.x_var + self._noise_var(snr))
        y = atoms.x_mean + sd * jax.random.normal(key, atoms.x_mean.shape)
        return {"snr": snr, "reading": jnp.where(snr > 0, y, 0.0)}

    def cell(self, payload):
        """Bin of each covered reading at ``edges``, cell 0 for an uncovered gas."""
        snr, y = payload["snr"], payload["reading"]
        code = jnp.searchsorted(self.edges, y, side="right")
        return _cell_index(jnp.where(snr > 0, code, 0), self.edges.shape[0] + 1)

    def n_cells(self, atoms):
        """``(len(edges) + 1) ** (K G)``."""
        n_planet, n_gas = atoms.x_mean.shape[-2:]
        return (self.edges.shape[0] + 1) ** (n_planet * n_gas)

    def gas_cells(self, atoms, action):
        """Each reading's bins, differences of normal CDFs; one for an unread gas."""
        snr = _read_snr(self.snr_one_day, atoms, action)
        covered = (snr > 0)[..., None]
        sd = jnp.sqrt(atoms.x_var + self._noise_var(snr))
        sd = jnp.where(snr > 0, sd, 1.0)[..., None]
        cdf = ndtr((self.edges - atoms.x_mean[..., None]) / sd)
        cdf = jnp.concatenate(
            [jnp.zeros_like(cdf[..., :1]), cdf, jnp.ones_like(cdf[..., :1])], axis=-1
        )
        n_bin = self.edges.shape[0] + 1
        return jnp.where(covered, jnp.diff(cdf, axis=-1), jnp.eye(n_bin)[0])

    def cells(self, atoms, action):
        """Closed form: the joint of the readings' bins (``gas_cells``)."""
        return _joint(self.gas_cells(atoms, action))


class StepReportModel(AbstractObservationModel):
    """A one-bit report per gas, given at or above S/N ``STEP_GATE_SNR``.

    Below the gate nothing is reported; at or above it, whether the gas is above its
    floor. The report is exact: at or above the gate the bit is ``x > GAS_FLOOR`` for
    the true abundance ``x``.

    The model applies only to point gas blocks (``x_var = 0``). On a Gaussian block two
    rows of one gas report the same bit, so their likelihoods are not independent and
    their sum would count the bit twice; the block after a bit is a truncated Gaussian,
    which ``Atoms`` cannot hold. ``sample`` and ``cells`` raise ``ValueError`` when
    the gas variance is a concrete nonzero value; a traced variance cannot be checked
    there, and the restriction holds all the same. ``stats`` raises at run time, under
    ``jit`` as well, for a row that reports on a gas whose variance is nonzero; a row
    that reports nothing reads no gas, so a model table holding this model replays
    Gaussian draws as long as none of its rows reports on them.

    Each gas has three outcome cells: 0 no report, 1 bit 0, 2 bit 1; a slot with no
    planet reports nothing. A row's cells are the joint over planets and gases, coded
    by ``cell`` and ``n_cells``. The report
    characterizes the planets it reads; it does not search (``searches`` is false).

    Attributes:
        snr_one_day: S/N of each gas after one day of exposure, shape ``(G,)``.
    """

    searches: ClassVar[bool] = False
    stage: ClassVar[int] = 0
    gas_step: ClassVar[float | None] = GAS_FLOOR  # the bit steps at the floor

    snr_one_day: Float[Array, " G"] = eqx.field(converter=jnp.asarray)

    def _check_point(self, atoms: Atoms) -> None:
        """Raise if the gas block is concretely not a point state."""
        try:
            spread = bool(jnp.any(atoms.x_var != 0))
        except jax.errors.ConcretizationTypeError:
            return  # traced: the docstring states the restriction
        if spread:
            raise ValueError("StepReportModel needs point gas draws (x_var = 0)")

    def _p_above(self, atoms, star) -> Float[Array, "A K G"]:
        """Probability of each draw's gas being above the floor."""
        n_gas = atoms.x_mean.shape[-1]
        probs = [GasAbove(g, GAS_FLOOR).prob(atoms, star) for g in range(n_gas)]
        return jnp.stack(probs, axis=-1)

    def stats(self, payload, atoms, star):
        """The row's reported bits, counted per planet and gas, ``(K, G)``.

        ``"above"`` is 1 where the gas reports bit 1, ``"below"`` 1 where it reports
        bit 0, both 0 for a silent gas. They do not depend on the draw, so a row costs
        the same at any number of draws.
        """
        snr, bit = payload["snr"], payload["bit"]
        reported = snr >= STEP_GATE_SNR
        up = bit > 0.5
        block = atoms.x_mean.shape[-2:]
        dtype = atoms.x_mean.dtype
        return {
            "above": jnp.broadcast_to(reported & up, block).astype(dtype),
            "below": jnp.broadcast_to(reported & ~up, block).astype(dtype),
        }

    def finalize(self, stats, atoms, star):
        """The summed log probabilities of the reported bits, with the draws unchanged.

        Rows are independent given a point draw, so their log probabilities add:
        ``n1 log P(x > floor) + n0 log P(x <= floor)`` per planet and gas, with a gas
        no row reported adding exactly zero. A bit about a point state changes nothing
        in it.

        Raises:
            EquinoxRuntimeError: if a reported gas is not a point state
                (``x_var != 0``).
        """
        above, below = stats["above"], stats["below"]
        p_above = self._p_above(atoms, star)

        def counted(n, p):
            some = n > 0
            return jnp.where(some, n * jnp.log(jnp.where(some, p, 1.0)), 0.0)

        log_p = (counted(above, p_above) + counted(below, 1.0 - p_above)).sum((-2, -1))
        log_p = eqx.error_if(
            log_p,
            jnp.any((above + below > 0) & (atoms.x_var != 0)),
            "StepReportModel needs point gas draws (x_var = 0) for a reported gas",
        )
        return log_p, atoms

    def sample(self, key, atoms, action):
        """Bits of every planet and gas, per draw; ``(A, K, G)`` arrays."""
        self._check_point(atoms)
        snr = _read_snr(self.snr_one_day, atoms, action)
        x = atoms.x_mean + jnp.sqrt(atoms.x_var) * jax.random.normal(
            key, atoms.x_mean.shape
        )
        bit = jnp.where(snr >= STEP_GATE_SNR, x > GAS_FLOOR, False)
        return {"snr": snr, "bit": bit.astype(snr.dtype)}

    def cell(self, payload):
        """0 for no report, otherwise 1 plus the bit."""
        snr, bit = payload["snr"], payload["bit"]
        code = jnp.where(snr >= STEP_GATE_SNR, 1 + (bit > 0.5), 0)
        return _cell_index(code, 3)

    def n_cells(self, atoms):
        """``3 ** (K G)``."""
        n_planet, n_gas = atoms.x_mean.shape[-2:]
        return 3 ** (n_planet * n_gas)

    def gas_cells(self, atoms, action):
        """Each gas's report: silent below the gate, else the bit above the floor."""
        self._check_point(atoms)
        snr = _read_snr(self.snr_one_day, atoms, action)
        p = self._p_above(atoms, ())
        reported = jnp.stack([jnp.zeros_like(p), 1.0 - p, p], axis=-1)
        silent = jnp.eye(3)[0]
        return jnp.where((snr >= STEP_GATE_SNR)[..., None], reported, silent)

    def cells(self, atoms, action):
        """Closed form: the joint of the reports (``gas_cells``)."""
        return _joint(self.gas_cells(atoms, action))


def _per_star(p: Float[Array, "*s"], star_index: Int[Array, ""]) -> Float[Array, ""]:
    """A per-star configuration value: a scalar shared by every star, or star's own."""
    return p if p.ndim == 0 else p[star_index]


def _detect(
    atoms: Atoms,
    exposure_d: Float[Array, ""],
    p_hz_one_day: Float[Array, ""],
    p_other_one_day: Float[Array, ""],
) -> Float[Array, "A K"]:
    """Probability that an exposure detects each planet slot of each draw.

    ``1 - (1 - p)**t`` with ``p`` the one-day probability of the planet's kind, zero
    for a slot with no planet and for no exposure; finite in every branch, so a zero
    payload (a padded or refused row) passes no NaN into a derivative.
    """
    p = jnp.where(atoms.cloud.samples["hz"] > 0.5, p_hz_one_day, p_other_one_day)
    sure = p >= 1.0
    log_miss = jnp.log1p(-jnp.where(sure, 0.0, p))
    looked = exposure_d > 0
    t = jnp.where(looked, exposure_d, 1.0)
    d = jnp.where(sure, 1.0, -jnp.expm1(t * log_miss))
    return jnp.where(atoms.exists & looked, d, 0.0)


class DetectionModel(AbstractObservationModel):
    """A search: an exposure detects each planet of a star independently.

    A planet in the habitable zone (the toy state's ``"hz"`` bit) is detected in one day
    of exposure with probability ``p_hz_one_day`` and any other planet with
    ``p_other_one_day``; an exposure of ``t`` days detects it with ``1 - (1 - p)**t``,
    smooth in the exposure (``t`` looks of one day). A slot with no planet is never
    detected. Each probability is a scalar shared by every star, or one per star read
    at the action's star (a per-target detection scale). With ``p = 1 - exp(-1 / tau)``
    a star's completeness is ``1 - exp(-t / tau)``.

    The payload holds the exposure, its start (``"time_d"``, the action's realized
    start), the two probabilities it was taken at and whether each planet slot was
    detected, ``"detected"`` of shape ``(K,)``, slots labeled as the truth's (declared
    deviation 2; ``detections`` reads them). Its cells are the joint of the planets'
    detections, ``2 ** K`` of them, mixed radix with the first planet most
    significant: cell 0 is the miss of every planet, and any other cell is a
    detection, which reaches stage 1 (a spectrum may follow it). Rows are independent
    given the draw, so ``stats`` is the log likelihood of each row per draw; it scores
    the detections of tracks only (``planet_tracks``): a detection in a slot no draw
    holds is of a planet new to the cloud, left for a refit. The toy's probabilities
    do not depend on the start; the payload keeps it for a model whose do.

    Attributes:
        p_hz_one_day: probability of detecting a planet in the habitable zone in one
            day of exposure, a scalar or one per star.
        p_other_one_day: the same for any other planet.
    """

    searches: ClassVar[bool] = True
    stage: ClassVar[int] = 1  # a detection opens a spectrum

    p_hz_one_day: Float[Array, "*s"] = eqx.field(converter=jnp.asarray)
    p_other_one_day: Float[Array, "*s"] = eqx.field(converter=jnp.asarray)

    def _probs(self, action: Action) -> tuple[Float[Array, ""], Float[Array, ""]]:
        """The two one-day probabilities at the action's star."""
        return (
            _per_star(self.p_hz_one_day, action.star),
            _per_star(self.p_other_one_day, action.star),
        )

    def sample(self, key, atoms, action):
        """Detections of every planet slot, per draw; ``(A, K)`` bits."""
        p_hz, p_other = self._probs(action)
        d = _detect(atoms, action.exposure_d, p_hz, p_other)
        detected = jax.random.uniform(key, d.shape) < d
        lead = d.shape[:1]
        return {
            "exposure_d": jnp.broadcast_to(action.exposure_d, lead),
            "time_d": jnp.broadcast_to(action.time_d, lead),
            "p_hz_one_day": jnp.broadcast_to(p_hz, lead),
            "p_other_one_day": jnp.broadcast_to(p_other, lead),
            "detected": detected.astype(d.dtype),
        }

    def stats(self, payload, atoms, star):
        """The log probability of the row's detections and misses under each draw.

        A detection in a slot no draw holds (a planet new to the cloud) is not scored:
        no draw can explain it, and a refit adds its track.
        """
        d = _detect(
            atoms,
            payload["exposure_d"],
            payload["p_hz_one_day"],
            payload["p_other_one_day"],
        )
        seen = payload["detected"] > 0.5
        scored = jnp.where(seen & ~planet_tracks(atoms), 1.0, jnp.where(seen, d, 1 - d))
        return jnp.sum(guarded_log(scored), axis=-1)

    def finalize(self, stats, atoms, star):
        """The summed log likelihoods, with the draws unchanged."""
        return stats, atoms

    def detections(self, payload):
        """The slots the row detected."""
        return payload["detected"] > 0.5

    def cell(self, payload):
        """The joint of the planets' detections, mixed radix, cell 0 the miss."""
        return _cell_index((payload["detected"] > 0.5).astype(int)[:, None], 2)

    def n_cells(self, atoms):
        """``2 ** K``."""
        return 2 ** atoms.exists.shape[-1]

    def cells(self, atoms, action):
        """Closed form: the joint of independent per-planet detections."""
        d = _detect(atoms, action.exposure_d, *self._probs(action))
        return _joint(jnp.stack([1.0 - d, d], axis=-1)[:, :, None, :])


class InHz(AbstractConjunct):
    """The planet is in the habitable zone: the toy state's ``"hz"`` bit."""

    def prob(self, atoms, star):
        """The ``"hz"`` bit of each draw and planet."""
        return atoms.cloud.samples["hz"]


class InSize(AbstractConjunct):
    """The planet is Earth-sized: the toy state's ``"size"`` bit."""

    def prob(self, atoms, star):
        """The ``"size"`` bit of each draw and planet."""
        return atoms.cloud.samples["size"]


def _log_bit(bit: Float[Array, "*s"], p: Float[Array, ""]) -> Float[Array, "*s"]:
    """``ln p`` where ``bit`` is set, ``ln(1 - p)`` where not, ``p`` shared."""
    return jnp.where(bit > 0.5, guarded_log(p), guarded_log(1.0 - p))


def point_atoms(hz, size, x, exists, prior: AbstractRolePrior) -> Atoms:
    """Point draws of the toy state, equal weights, on ``prior``'s chart.

    Args:
        hz: the ``"hz"`` bits, shape ``(A, K)``.
        size: the ``"size"`` bits, shape ``(A, K)``.
        x: the gas abundances, dex, shape ``(A, K, G)``; a point block.
        exists: whether each slot holds a planet, shape ``(A, K)``.
        prior: the refit's prior the draws are read against.

    Returns:
        The draws: log weights zero, a placeholder target of zero (the toy draws are
        built, not fitted) and an unknown evidence.
    """
    n_draw = jnp.shape(hz)[0]
    cloud = WeightedCloud(
        {"hz": hz, "size": size},
        jnp.zeros(n_draw),
        jnp.zeros(n_draw),
        jnp.nan,
        prior,
    )
    return Atoms(cloud, x, jnp.zeros_like(x), exists)


class ToyPrior(AbstractRolePrior):
    """A role prior over whole draws of the toy state, each planet a point state.

    Slot 0 holds a planet with probability ``p_exist[0]``. The other slots, the free
    slots, hold the undetected planets by ``pool_law``: under ``BERNOULLI`` (the
    default) slot ``k`` holds a planet with probability ``p_exist[k]``,
    independently, so one free slot is the Bernoulli pool law, at most one planet;
    under ``POISSON`` the free slots hold ``N`` planets, ``N`` Poisson of mean
    ``sum(p_exist[1:])`` with the mass at ``N >= K - 1`` lumped at ``K - 1`` (the
    Poisson pool law truncated at the free slots), in a subset of the free slots drawn
    uniformly: the slots are exchangeable, so which slot a search detects says nothing
    about the planets in the others beyond their count. A planet is in the
    habitable zone with probability ``hz_scale`` times the star's ``"p_hz"`` (clipped
    to ``[0, 1]``), Earth-sized with ``p_size``, and each of its ``n_gas`` gases is
    ``N(gas_mean_dex, gas_sd_dex**2)``, independently. Draws with fewer slots than
    ``p_exist`` (the marks of a pool, one planet each) read the first slots' terms.
    The star is a pytree holding ``"p_hz"``.

    Attributes:
        p_exist: each slot's probability of holding a planet, shape ``(K,)``; under
            ``POISSON`` the free slots' entries sum to the mean count.
        hz_scale: the scale of the star's habitable-zone probability.
        p_size: the probability that a planet is Earth-sized.
        gas_mean_dex: the mean of each gas, dex.
        gas_sd_dex: the spread of each gas, dex.
        n_gas: the number of gases, static.
        pool_law: the law of the free slots' planets, ``BERNOULLI`` or ``POISSON``
            (``Pool.law``), static.
    """

    p_exist: Float[Array, " K"] = eqx.field(converter=jnp.asarray)
    hz_scale: Float[Array, ""] = eqx.field(converter=jnp.asarray)
    p_size: Float[Array, ""] = eqx.field(converter=jnp.asarray)
    gas_mean_dex: Float[Array, ""] = eqx.field(converter=jnp.asarray)
    gas_sd_dex: Float[Array, ""] = eqx.field(converter=jnp.asarray)
    n_gas: int = eqx.field(static=True)
    pool_law: str = eqx.field(static=True, default=BERNOULLI)

    def __check_init__(self):
        """Check the free slots' law."""
        if self.pool_law not in (BERNOULLI, POISSON):
            raise ValueError(f"the pool law is {BERNOULLI!r} or {POISSON!r}")

    def _p_hz(self, star) -> Float[Array, ""]:
        """The habitable-zone probability at the star."""
        return jnp.clip(star["p_hz"] * self.hz_scale, 0.0, 1.0)

    def log_prob(self, atoms, star):
        """The log density of each draw: its slots' existence, bits and gases."""
        n_slot = atoms.exists.shape[-1]
        p_exist = self.p_exist[..., :n_slot]
        samples = atoms.cloud.samples
        planet = (
            _log_bit(samples["hz"], self._p_hz(star))
            + _log_bit(samples["size"], self.p_size)
            + norm.logpdf(atoms.x_mean, self.gas_mean_dex, self.gas_sd_dex).sum(-1)
        )
        slot = jnp.where(
            atoms.exists, guarded_log(p_exist) + planet, guarded_log(1.0 - p_exist)
        )
        if self.pool_law == BERNOULLI or n_slot == 1:  # static
            return slot.sum(-1)
        # Poisson: the free slots' count, lumped at K - 1, over its equally likely
        # subsets of the free slots.
        free = atoms.exists[..., 1:]
        n_free = n_slot - 1
        count = jnp.sum(free, axis=-1)
        mean = jnp.sum(self.p_exist[1:])
        log_mean = guarded_log(mean)
        head = -mean + jnp.where(count > 0, count * log_mean, 0.0) - gammaln(count + 1)
        tail = guarded_log(gammainc(jnp.asarray(n_free, mean.dtype), mean))
        log_count = jnp.where(count < n_free, head, tail)
        log_subsets = (
            gammaln(n_free + 1.0) - gammaln(count + 1.0) - gammaln(n_free - count + 1.0)
        )
        planets = jnp.where(free, planet[..., 1:], 0.0).sum(-1)
        return slot[..., 0] + log_count - log_subsets + planets

    def sample(self, key, star):
        """One point draw of the star (``A = 1``)."""
        return self.draws(key, star, 1)

    def draws(self, key, star, n_draw: int) -> Atoms:
        """``n_draw`` point draws of the star, equal weights, on this prior.

        Args:
            key: the key of the draws.
            star: the star, a pytree holding ``"p_hz"``.
            n_draw: the number of draws, static.

        Returns:
            The draws, on this prior.
        """
        k_exist, k_hz, k_size, k_gas = jax.random.split(key, 4)
        shape = (n_draw, self.p_exist.shape[-1])
        exists = jax.random.bernoulli(k_exist, self.p_exist, shape)
        if self.pool_law == POISSON and shape[1] > 1:  # static
            mean = jnp.sum(self.p_exist[1:])
            count = jax.random.poisson(jax.random.fold_in(k_exist, 1), mean, shape[:1])
            order = jax.random.uniform(
                jax.random.fold_in(k_exist, 2), (n_draw, shape[1] - 1)
            )
            free = jnp.argsort(jnp.argsort(order, axis=-1), axis=-1) < count[:, None]
            exists = jnp.concatenate([exists[:, :1], free], axis=-1)
        hz = jax.random.bernoulli(k_hz, self._p_hz(star), shape) & exists
        size = jax.random.bernoulli(k_size, self.p_size, shape) & exists
        x = self.gas_mean_dex + self.gas_sd_dex * jax.random.normal(
            k_gas, (*shape, self.n_gas)
        )
        x = jnp.where(exists[..., None], x, 0.0)
        return point_atoms(hz.astype(x.dtype), size.astype(x.dtype), x, exists, self)


class ToyRefitter(AbstractRefitter):
    """An importance refit from fresh draws of the law of one planet, rows rescored.

    The new cloud tracks the old cloud's planets and each slot the term's rows detected
    that no draw holds (``planet_tracks``): every draw holds a planet in each of those
    slots, its state drawn afresh from ``law`` (the law of one planet, the pool marks'
    law), and no planet elsewhere. Each of the ``A`` draws is chosen from
    ``proposal_factor`` candidates, all drawn from the law, in proportion to the joint
    likelihood of every row the new refit represents (those the old refit represented
    and the term's), scored through the models' statistics (``replay_likelihood``),
    and its weight is the mean of the candidates' likelihoods. That pair is properly
    weighted for the prior times the likelihood (the multiple-try identity
    ``E[mean_j L_j f(y_chosen)] = E[L f]``), so more candidates lower the variance of
    the weights, never their mean: a refused refit retried with more candidates draws
    a cloud of the same ``A`` with a larger Kish size. Every slot's state is drawn
    from the law of one planet, which is the refit prior's law of a planet in a slot
    it holds, and a slot's existence is the same in every draw, so the prior over the
    proposal is constant and the weights are the likelihood alone. The refit never
    keeps an old weight or state, so it holds for any cloud. Weights the gate refuses
    (``gate_weights``) are NaN. The cloud's target and evidence are placeholders, as
    ``point_atoms``'.

    Attributes:
        law: the law of one planet, a one-slot ``ToyPrior``; its existence
            probability is not read (a track exists).
    """

    law: ToyPrior

    def refit(self, refit, term, key, *, min_n_eff, max_khat, proposal_factor=1):
        """The star's cloud redrawn with its tracks and each new planet, rescored."""
        atoms = refit.atoms
        n_draw, n_slot = atoms.exists.shape
        n_try = int(proposal_factor)
        detected = detected_slots(term.log, term.rows, term.models, n_slot)
        tracks = planet_tracks(atoms) | detected  # (K,)
        k_draw, k_pick = jax.random.split(key)
        law = eqx.tree_at(lambda p: p.p_exist, self.law, jnp.ones(1))
        fresh = law.draws(k_draw, term.star, n_draw * n_try * n_slot)
        n_prop = n_draw * n_try
        exists = jnp.broadcast_to(tracks, (n_prop, n_slot))

        def slots(x):
            x = x.reshape(n_prop, n_slot, *x.shape[2:])
            return jnp.where(exists.reshape(*exists.shape, *([1] * (x.ndim - 2))), x, 0)

        samples = fresh.cloud.samples
        cloud = WeightedCloud(
            {"hz": slots(samples["hz"]), "size": slots(samples["size"])},
            jnp.zeros(n_prop),
            jnp.zeros(n_prop),
            jnp.nan,
            atoms.cloud.prior,
        )
        x = slots(fresh.x_mean)
        proposed = Atoms(cloud, x, jnp.zeros_like(x), exists)
        represented = refit.row_mask | term.rows
        # Every represented row is released, so the replay applies them all.
        scored = Refit(proposed, ~represented, refit.star_index)
        every = jnp.asarray(jnp.inf)
        rescored = replay(scored, term.log, term.models, term.star, every)
        drawn, log_like = replay_likelihood(rescored, term.models, term.star)
        log_like = log_like.reshape(n_draw, n_try)
        log_w = logsumexp(log_like, axis=-1) - jnp.log(n_try)
        pick = jax.random.categorical(k_pick, log_like, axis=-1)  # (A,)
        chosen = jnp.arange(n_draw) * n_try + pick

        def take(leaf):
            return leaf[chosen]

        log_w = gate_weights(log_w, min_n_eff=min_n_eff, max_khat=max_khat)
        cloud = WeightedCloud(
            jax.tree.map(take, drawn.cloud.samples),
            log_w,
            jnp.zeros(n_draw),
            jnp.nan,
            atoms.cloud.prior,
        )
        weighted = Atoms(
            cloud, take(drawn.x_mean), take(drawn.x_var), take(drawn.exists)
        )
        return Refit(weighted, represented, refit.star_index)

    def add_tracks(self, refit, slots, key, star):
        """Each draw with a planet in each of ``slots``, its state from ``law``."""
        atoms = refit.atoms
        n_draw, n_slot = atoms.exists.shape
        law = eqx.tree_at(lambda p: p.p_exist, self.law, jnp.ones(1))
        fresh = law.draws(key, star, n_draw * n_slot)

        def place(old, drawn):
            drawn = drawn.reshape(n_draw, n_slot, *drawn.shape[2:])
            return jnp.where(slots.reshape(1, -1, *([1] * (old.ndim - 2))), drawn, old)

        samples = atoms.cloud.samples
        samples = {
            "hz": place(samples["hz"], fresh.cloud.samples["hz"]),
            "size": place(samples["size"], fresh.cloud.samples["size"]),
        }
        cloud = eqx.tree_at(lambda c: c.samples, atoms.cloud, samples)
        added = Atoms(
            cloud,
            place(atoms.x_mean, fresh.x_mean),
            place(atoms.x_var, fresh.x_var),
            atoms.exists | slots,
        )
        return Refit(added, refit.row_mask, refit.star_index)


class ToySurvey(eqx.Module):
    """A toy study with what a run of it reads besides a policy.

    Attributes:
        study: the study: claim, truth, roles, each star's refit and pool, the stars,
            the design and the refitter.
        menu: the planner's menu of each star.
        grant: the rows given at time zero.
    """

    study: Study
    menu: Menu
    grant: Grant


TOY_SEARCH_D = (0.5, 1.0)
"""Exposures of the toy survey's two search entries, days."""

TOY_SPECTRUM_D = (0.6, 1.2)
"""Exposures of the toy survey's two spectrum entries, days."""

TOY_LATENCY_D = (2.0, 10.0)
"""Latency of the toy survey's search and spectrum modes, days."""

TOY_KEEPOUT_D = 120.0
"""Days of each year a toy star spends in keepout, one window a year."""

TOY_YEAR_D = 365.25
"""The toy's year, days: the period of its keepout."""

TOY_KEEPOUT_STEP_D = 5.0
"""Width of a bin of the toy survey's keepout table, days."""


def toy_keepout(key, n_star: int, end_d: float) -> Keepout:
    """A keepout table: each star closed ``TOY_KEEPOUT_D`` days a year.

    Star ``s`` is closed while the time since its window opened, modulo the year, is
    below ``TOY_KEEPOUT_D``; the windows open at days uniform over the year, read at
    the middle of each bin of ``TOY_KEEPOUT_STEP_D`` days over ``[0, end_d)``.

    Args:
        key: the key of the windows.
        n_star: the number of stars, ``S``.
        end_d: the calendar's end, days.

    Returns:
        The table.
    """
    opens_d = jax.random.uniform(key, (n_star, 1), maxval=TOY_YEAR_D)
    n_bin = max(1, int(-(-end_d // TOY_KEEPOUT_STEP_D)))
    middle_d = (jnp.arange(n_bin) + 0.5) * TOY_KEEPOUT_STEP_D
    closed = jnp.mod(middle_d - opens_d, TOY_YEAR_D) < TOY_KEEPOUT_D
    return Keepout(closed, TOY_KEEPOUT_STEP_D)


def toy_survey(
    key,
    n_star: int,
    n_draw: int,
    n_mark: int,
    *,
    pool_mass: float,
    budget_d: float,
    end_d: float,
    capacity: int,
    n_slot: int = 2,
    n_grant: int = 0,
    pool_law: str = BERNOULLI,
) -> ToySurvey:
    """A toy survey of many stars, built in one place for the tests and the scripts.

    Each star has a known planet, the track in slot 0, and ``pool_mass`` undetected
    planets in expectation, which the truth holds in the free slots ``1 .. n_slot -
    1`` by the study's pool law (``pool_law``), the planner's own: under
    ``BERNOULLI`` (one free slot) a planet with probability ``pool_mass``, at most
    one; under ``POISSON`` a Poisson count of mean ``pool_mass``, lumped at the
    ``n_slot - 1`` free slots, in exchangeable slots (``ToyPrior``). The pool's law is
    the truth's, so the study is well specified: the planner's forecast of the pool is
    the law the worlds are drawn from, up to the marks' Monte Carlo error and, under
    ``POISSON``, the lumped mass ``P(N > n_slot - 1)``. A planet is in the habitable
    zone with the star's ``"p_hz"`` (uniform on ``[0.2, 0.6]`` over the
    stars), Earth-sized half the time, and has one gas ``N(GAS_FLOOR,
    GAS_PRIOR_SD**2)``. That is the truth's prior (``ToyPrior``), from which the worlds
    are drawn. Each star's refit holds ``n_draw`` point draws of equal weight of the
    track alone (the free slots empty: the pool holds the undetected planets), on the
    truth's prior. The pool holds ``pool_mass`` planets, every role, on ``n_mark``
    marks drawn from the truth's prior of one planet, its role prior ratios formed
    ungated (``role_pool``; a planner gates them again). The forecast is the truth; the
    verdict has two members, one with 0.8 times the habitable-zone probability and
    gases ``N(0, 0.8**2)``, one with 1.2 times it and gases ``N(0.2, 1.2**2)``, the
    free slots' existence as the truth's. The claim is a planet in the habitable zone
    with its gas above the floor.

    The design has a search (``DetectionModel``: a planet in the habitable zone is
    found in one day with probability 0.75, any other with 0.5) and a spectrum
    (``GaussianGasModel`` at S/N 4 in one day), released ``TOY_LATENCY_D`` after their
    exposures; a calendar that ends at ``end_d``; and a keepout table
    (``toy_keepout``). The menu offers each star two searches (``TOY_SEARCH_D``) and
    two spectra (``TOY_SPECTRUM_D``), the spectra once a search has found a planet
    (stage 1). The grant is one archival search (the first exposure) of each of the
    first ``n_grant`` stars. A search that finds a pool planet hands it to the
    study's refitter (``ToyRefitter``, on the law of one planet).

    Args:
        key: the key of the stars, the refits, the marks and the keepout.
        n_star: the number of stars, ``S``.
        n_draw: the draws of each star's refit, ``A``.
        n_mark: the marks of each star's pool.
        pool_mass: the expected number of undetected planets at each star; below 1
            under ``BERNOULLI`` (a free slot certain to hold a planet would leave
            every refit draw, whose free slots are empty, outside the prior).
        budget_d: the survey's budget, days.
        end_d: the calendar's end, days.
        capacity: the log's row capacity ``W``.
        n_slot: the planet capacity of a star, ``K``: the track and the free slots;
            2 under ``BERNOULLI``.
        n_grant: the number of stars with an archival search.
        pool_law: the law of the undetected planets, the pool's and the truth's
            (``Pool.law``), static.

    Returns:
        The survey.

    Raises:
        ValueError: if the free slots cannot hold the pool's law.
    """
    bernoulli = pool_law == BERNOULLI
    if (
        n_slot < 2
        or pool_mass < 0.0
        or (bernoulli and not (n_slot == 2 and pool_mass < 1.0))
    ):
        raise ValueError(
            f"{n_slot - 1} free slots cannot hold a {pool_law} pool of mass {pool_mass}"
        )
    p_exist = [1.0] + [pool_mass / (n_slot - 1)] * (n_slot - 1)
    law = {"n_gas": 1, "pool_law": pool_law}
    truth = ToyPrior("toy", p_exist, 1.0, 0.5, GAS_FLOOR, GAS_PRIOR_SD, **law)
    members = (
        ToyPrior("toy", p_exist, 0.8, 0.5, GAS_FLOOR, 0.8 * GAS_PRIOR_SD, **law),
        ToyPrior("toy", p_exist, 1.2, 0.5, GAS_FLOOR + 0.2, 1.2 * GAS_PRIOR_SD, **law),
    )
    verdict = jax.tree.map(lambda *x: jnp.stack(x), *members)
    roles = Roles(truth, verdict)
    claim = Claim((InHz(), GasAbove(0, GAS_FLOOR)))

    k_star, k_draw, k_mark, k_keepout = jax.random.split(key, 4)
    stars = {"p_hz": jax.random.uniform(k_star, (n_star,), minval=0.2, maxval=0.6)}
    one_planet = eqx.tree_at(lambda p: p.p_exist, truth, truth.p_exist[:1])
    track = eqx.tree_at(lambda p: p.p_exist, truth, truth.p_exist.at[1:].set(0.0))
    n_role = 3

    def per_star(k_d, k_m, star, index):
        drawn = track.draws(k_d, star, n_draw)
        atoms = eqx.tree_at(lambda a: a.cloud.prior, drawn, truth)
        refit = Refit(atoms, jnp.zeros(capacity, bool), index)
        drawn = one_planet.draws(k_m, star, n_mark)
        marks = eqx.tree_at(lambda a: a.cloud.prior, drawn, truth)
        pool = role_pool(
            marks,
            roles,
            jnp.zeros((n_role, n_mark)),
            jnp.full(n_role, jnp.log(pool_mass)),
            star,
            min_n_eff=0.0,
            max_khat=float("inf"),
            law=pool_law,
        )
        return refit, pool

    refits, pools = jax.vmap(per_star)(
        jax.random.split(k_draw, n_star),
        jax.random.split(k_mark, n_star),
        stars,
        jnp.arange(n_star),
    )

    models = (
        DetectionModel(p_hz_one_day=0.75, p_other_one_day=0.5),
        GaussianGasModel(snr_one_day=[4.0]),
    )
    modes = (Mode(0, TOY_LATENCY_D[0]), Mode(1, TOY_LATENCY_D[1]))
    keepout = toy_keepout(k_keepout, n_star, end_d)
    design = Design(models, modes, Resources(budget_d, end_d), keepout, capacity)
    study = Study(
        claim,
        truth,
        truth,
        verdict,
        refits,
        pools,
        stars,
        design,
        jnp.zeros((n_star, n_role)),
        alpha=0.05,
        refitter=ToyRefitter(one_planet),
    )

    exposures = (*TOY_SEARCH_D, *TOY_SPECTRUM_D)
    menu = Menu(
        jnp.broadcast_to(jnp.asarray(exposures), (n_star, len(exposures))),
        jnp.ones((n_star, len(exposures)), bool),
        (0, 0, 1, 1),
        (0, 0, 1, 1),
    )
    grant = Grant(
        Action(
            jnp.arange(n_grant),
            jnp.zeros(n_grant, int),
            jnp.full(n_grant, TOY_SEARCH_D[0]),
        )
    )
    return ToySurvey(study, menu, grant)
