"""Toy observation models: two gas models on the gas block.

An observation model (``AbstractObservationModel``, defined in ``types``) turns an
action on a star into a row payload (``sample``), scores rows against every draw through
additive statistics (``stats``, ``finalize``), and gives the distribution of a row's
outcome over discrete cells (``cells``). The host passes the model; nothing dispatches
on a traced model id.

Both toy models read the gas block of ``Atoms`` and write one payload per row, a dict of
named ``(K, G)`` arrays: ``"snr"``, the S/N reached by every planet and gas (zero for a
gas the mode does not cover), and the reading (``"reading"`` for the Gaussian model,
``"bit"`` for the step report).

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
from jax.scipy.special import ndtr
from jaxtyping import Array, Float, Int

from spaceodyssey.claim import GasAbove
from spaceodyssey.types import AbstractObservationModel, Action, Atoms

GAS_FLOOR = 0.0
"""Abundance floor of every gas in the toy, dex (a toy constant)."""

GAS_PRIOR_SD = 1.0
"""Width of the verdict's gas prior at the floor, dex (a toy constant)."""

READING_SD_AT_UNIT_SNR = 2.5
"""Noise of a gas reading at S/N 1, dex: ``sigma = 2.5 / (S/N)``."""

STEP_GATE_SNR = 5.0
"""S/N at or above which the step report gives its bit."""


def _snr(snr_one_day: Float[Array, " G"], action: Action) -> Float[Array, " G"]:
    """S/N per gas reached by an action: it grows as the root of the exposure."""
    return snr_one_day * jnp.sqrt(action.exposure_d)


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
    (cell 0). A row's cells are the joint over planets and gases; ``cell`` and
    ``n_cells`` give the same coding for ``binned_cells``.

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

    snr_one_day: Float[Array, " G"] = eqx.field(converter=jnp.asarray)
    edges: Float[Array, " E"] = eqx.field(converter=jnp.asarray, default=(GAS_FLOOR,))

    def _noise_var(self, snr: Float[Array, "*s"]) -> Float[Array, "*s"]:
        """Reading variance; zero where the gas is not covered (no reading)."""
        safe = jnp.where(snr > 0, snr, 1.0)
        return jnp.where(snr > 0, (READING_SD_AT_UNIT_SNR / safe) ** 2, 0.0)

    def _precision(self, snr: Float[Array, "*s"]) -> Float[Array, "*s"]:
        """Reading precision ``1 / sigma**2``; zero where the gas is not covered."""
        return jnp.where(snr > 0, (snr / READING_SD_AT_UNIT_SNR) ** 2, 0.0)

    def stats(self, payload, atoms, star):
        """Precision-weighted sums of the readings of one row, per planet and gas.

        With precision ``p = 1 / sigma**2`` (zero for an uncovered gas) and reading
        ``y``: ``"precision"`` is ``p``, ``"weighted_sum"`` ``p y``,
        ``"weighted_square_sum"`` ``p y**2`` and ``"log_norm"`` the normalizer
        ``log(p / (2 pi)) / 2`` (zero for an uncovered gas). They do not depend on the
        draw and are broadcast to the gas block, ``(A, K, G)``.
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
        return jax.tree.map(lambda x: jnp.broadcast_to(x, atoms.x_mean.shape), per_gas)

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
        safe = jnp.where(total > 0, total, 1.0)
        ybar = jnp.where(total > 0, stats["weighted_sum"] / safe, 0.0)
        scatter = stats["weighted_square_sum"] - stats["weighted_sum"] * ybar
        shrink = 1.0 / (1.0 + v * total)
        residual = ybar - m
        terms = stats["log_norm"] - 0.5 * (
            scatter + total * residual**2 * shrink - jnp.log(shrink)
        )
        mean = m + v * total * residual * shrink
        updated = eqx.tree_at(lambda a: (a.x_mean, a.x_var), atoms, (mean, v * shrink))
        return terms.sum(axis=(1, 2)), updated

    def sample(self, atoms, action, key):
        """Readings of every planet and gas, per draw; ``(A, K, G)`` arrays."""
        snr = jnp.broadcast_to(_snr(self.snr_one_day, action), atoms.x_mean.shape)
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

    def cells(self, atoms, action):
        """Closed form: each bin is a difference of normal CDFs of the reading."""
        snr = jnp.broadcast_to(_snr(self.snr_one_day, action), atoms.x_mean.shape)
        covered = (snr > 0)[..., None]
        sd = jnp.sqrt(atoms.x_var + self._noise_var(snr))
        sd = jnp.where(snr > 0, sd, 1.0)[..., None]
        cdf = ndtr((self.edges - atoms.x_mean[..., None]) / sd)
        cdf = jnp.concatenate(
            [jnp.zeros_like(cdf[..., :1]), cdf, jnp.ones_like(cdf[..., :1])], axis=-1
        )
        n_bin = self.edges.shape[0] + 1
        factors = jnp.where(covered, jnp.diff(cdf, axis=-1), jnp.eye(n_bin)[0])
        return _joint(factors)


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

    Each gas has three outcome cells: 0 no report, 1 bit 0, 2 bit 1; a row's cells are
    the joint over planets and gases, coded by ``cell`` and ``n_cells``. The report
    characterizes the planets it reads; it does not search (``searches`` is false).

    Attributes:
        snr_one_day: S/N of each gas after one day of exposure, shape ``(G,)``.
    """

    searches: ClassVar[bool] = False

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
        """The row's log probability of the reported bits; a silent gas adds zero."""
        snr, bit = payload["snr"], payload["bit"]
        reported = snr >= STEP_GATE_SNR
        p_above = self._p_above(atoms, star)
        p = jnp.where(bit > 0.5, p_above, 1.0 - p_above)
        log_p = jnp.where(reported, jnp.log(p), 0.0).sum(axis=(1, 2))
        return eqx.error_if(
            log_p,
            jnp.any(reported & (atoms.x_var != 0)),
            "StepReportModel needs point gas draws (x_var = 0) for a reported gas",
        )

    def finalize(self, stats, atoms, star):
        """The summed log probabilities, with the draws unchanged.

        Rows are independent given a point draw, so their log probabilities add, and
        a bit about a point state changes nothing in it. ``stats`` has checked the
        draws of every reporting row.
        """
        return stats, atoms

    def sample(self, atoms, action, key):
        """Bits of every planet and gas, per draw; ``(A, K, G)`` arrays."""
        self._check_point(atoms)
        snr = jnp.broadcast_to(_snr(self.snr_one_day, action), atoms.x_mean.shape)
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

    def cells(self, atoms, action):
        """Closed form: the bit follows the probability of the gas above its floor."""
        self._check_point(atoms)
        snr = jnp.broadcast_to(_snr(self.snr_one_day, action), atoms.x_mean.shape)
        p = self._p_above(atoms, ())
        reported = jnp.stack([jnp.zeros_like(p), 1.0 - p, p], axis=-1)
        silent = jnp.eye(3)[0]
        return _joint(jnp.where((snr >= STEP_GATE_SNR)[..., None], reported, silent))
