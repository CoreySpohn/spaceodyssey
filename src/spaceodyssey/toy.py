"""Toy observation laws: two gas laws on the gas block.

A law (``AbstractLaw``, defined in ``types``) turns an action on a star into a row
payload (``sample``), scores a payload against every draw (``log_like``), and gives the
distribution of a row's outcome over discrete cells (``cells``). The host passes the
law module; nothing dispatches on a traced law id.

Both toy laws read the gas block of ``Atoms`` and write one payload of shape
``(K, G, 2)`` per row: for every planet and gas, the S/N reached (zero for a gas the
mode does not cover) and the reading.

Toy constants. The floor, the width of the verdict's gas prior and the linear shape of
the reading are proposed for the toy and wait on approval:

- ``GAS_FLOOR``: 0 dex. Abundance is measured as height above the floor.
- ``GAS_PRIOR_SD``: 1 dex. The verdict's gas prior is ``N(GAS_FLOOR, GAS_PRIOR_SD**2)``,
  so a gas starts at even odds of being above its floor.
- The reading is linear at every S/N; there is no separate shape below detection.
- These laws are never used for a negative certificate.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.special import ndtr
from jaxtyping import Array, Float, Int

from spaceodyssey.claim import GasAbove
from spaceodyssey.types import AbstractLaw, Action, Atoms

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


def _pack(snr: Float[Array, "*A K G"], value: Float[Array, "*A K G"]) -> Array:
    """Payload ``(..., K, G, 2)``: the S/N and the reading of each planet and gas."""
    return jnp.stack([snr, value], axis=-1)


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


class GaussianGasLaw(AbstractLaw):
    """A linear Gaussian reading of every gas abundance.

    The reading is ``y = x + eps`` with ``eps ~ N(0, sigma**2)`` and
    ``sigma = READING_SD_AT_UNIT_SNR / (S/N)`` dex: no gravity slope, no bias, and the
    same linear shape at every S/N. Noise is independent across planets and gases. A gas
    with S/N 0 is not covered: its reading is 0 and carries no information.

    The outcome cells bin each reading at ``edges``; an uncovered gas has one outcome
    (cell 0). A row's cells are the joint over planets and gases.

    Attributes:
        snr_one_day: S/N of each gas after one day of exposure, shape ``(G,)``.
        edges: bin edges of a reading, dex, increasing; default one edge at the floor.
    """

    snr_one_day: Float[Array, " G"] = eqx.field(converter=jnp.asarray)
    edges: Float[Array, " E"] = eqx.field(converter=jnp.asarray, default=(GAS_FLOOR,))

    def _noise_var(self, snr: Float[Array, "*s"]) -> Float[Array, "*s"]:
        """Reading variance; zero where the gas is not covered (no reading)."""
        safe = jnp.where(snr > 0, snr, 1.0)
        return jnp.where(snr > 0, (READING_SD_AT_UNIT_SNR / safe) ** 2, 0.0)

    def log_like(self, payload, atoms, star):
        """Log density of the readings, with each draw's gas Gaussian integrated out.

        For a draw with mean ``m`` and variance ``v`` the reading is
        ``N(y | m, v + sigma**2)``; a point state (``v = 0``) gives ``N(y | x,
        sigma**2)``. Uncovered gases contribute zero.

        This is the marginal of one row. Two rows of one gas on a draw with ``v > 0``
        share the draw's abundance, so the sum of their marginals is not their joint
        likelihood; a sum over rows is exact for point states, or for at most one row
        per gas. ``update`` folds rows into the block one at a time.
        """
        snr, y = payload[..., 0], payload[..., 1]  # (K, G)
        var = atoms.x_var + self._noise_var(snr)
        safe = jnp.where(snr > 0, var, 1.0)
        terms = -0.5 * ((y - atoms.x_mean) ** 2 / safe + jnp.log(2 * jnp.pi * safe))
        return jnp.where(snr > 0, terms, 0.0).sum(axis=(1, 2))

    def update(self, payload: Array, atoms: Atoms) -> Atoms:
        """Conjugate update of every draw's gas Gaussian by one row.

        With prior ``N(m, v)`` and a reading ``y`` of variance ``sigma**2``, the gain is
        ``v / (v + sigma**2)``, the mean becomes ``m + gain * (y - m)`` and the variance
        ``v * (1 - gain)``. An uncovered gas and a point state are unchanged.

        Args:
            payload: the payload of one row, ``(K, G, 2)``.
            atoms: the draws.

        Returns:
            The draws with their gas block updated.
        """
        snr, y = payload[..., 0], payload[..., 1]
        precision = jnp.where(snr > 0, (snr / READING_SD_AT_UNIT_SNR) ** 2, 0.0)
        gain = atoms.x_var * precision / (1.0 + atoms.x_var * precision)
        mean = atoms.x_mean + gain * (y - atoms.x_mean)
        var = atoms.x_var * (1.0 - gain)
        return eqx.tree_at(lambda a: (a.x_mean, a.x_var), atoms, (mean, var))

    def sample(self, atoms, action, key):
        """Readings of every planet and gas, per draw; shape ``(A, K, G, 2)``."""
        snr = jnp.broadcast_to(_snr(self.snr_one_day, action), atoms.x_mean.shape)
        sd = jnp.sqrt(atoms.x_var + self._noise_var(snr))
        y = atoms.x_mean + sd * jax.random.normal(key, atoms.x_mean.shape)
        return _pack(snr, jnp.where(snr > 0, y, 0.0))

    def cell(self, payload):
        """Bin of each covered reading at ``edges``, cell 0 for an uncovered gas."""
        snr, y = payload[..., 0], payload[..., 1]
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


class StepReportLaw(AbstractLaw):
    """A one-bit report per gas, given at or above S/N ``STEP_GATE_SNR``.

    Below the gate nothing is reported; at or above it, whether the gas is above its
    floor. The report is exact: at or above the gate the bit is ``x > GAS_FLOOR`` for
    the true abundance ``x``.

    The law applies only to point gas blocks (``x_var = 0``). On a Gaussian block two
    rows of one gas report the same bit, so their likelihoods are not independent and
    their sum would count the bit twice; the block after a bit is a truncated Gaussian,
    which ``Atoms`` cannot hold. Every method raises ``ValueError`` when the gas
    variance is a concrete nonzero value; a traced variance cannot be checked, and the
    restriction holds all the same.

    Each gas has three outcome cells: 0 no report, 1 bit 0, 2 bit 1; a row's cells are
    the joint over planets and gases.

    Attributes:
        snr_one_day: S/N of each gas after one day of exposure, shape ``(G,)``.
    """

    snr_one_day: Float[Array, " G"] = eqx.field(converter=jnp.asarray)

    def _check_point(self, atoms: Atoms) -> None:
        """Raise if the gas block is concretely not a point state."""
        try:
            spread = bool(jnp.any(atoms.x_var != 0))
        except jax.errors.ConcretizationTypeError:
            return  # traced: the docstring states the restriction
        if spread:
            raise ValueError("StepReportLaw needs point gas draws (x_var = 0)")

    def _p_above(self, atoms, star) -> Float[Array, "A K G"]:
        """Probability of each draw's gas being above the floor."""
        n_gas = atoms.x_mean.shape[-1]
        probs = [GasAbove(g, GAS_FLOOR).prob(atoms, star) for g in range(n_gas)]
        return jnp.stack(probs, axis=-1)

    def log_like(self, payload, atoms, star):
        """Log probability of the reported bits; a gas below the gate adds zero."""
        self._check_point(atoms)
        snr, bit = payload[..., 0], payload[..., 1]
        p_above = self._p_above(atoms, star)
        p = jnp.where(bit > 0.5, p_above, 1.0 - p_above)
        return jnp.where(snr >= STEP_GATE_SNR, jnp.log(p), 0.0).sum(axis=(1, 2))

    def sample(self, atoms, action, key):
        """Bits of every planet and gas, per draw; shape ``(A, K, G, 2)``."""
        self._check_point(atoms)
        snr = jnp.broadcast_to(_snr(self.snr_one_day, action), atoms.x_mean.shape)
        x = atoms.x_mean + jnp.sqrt(atoms.x_var) * jax.random.normal(
            key, atoms.x_mean.shape
        )
        bit = jnp.where(snr >= STEP_GATE_SNR, x > GAS_FLOOR, False)
        return _pack(snr, bit.astype(snr.dtype))

    def cell(self, payload):
        """0 for no report, otherwise 1 plus the bit."""
        snr, bit = payload[..., 0], payload[..., 1]
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
