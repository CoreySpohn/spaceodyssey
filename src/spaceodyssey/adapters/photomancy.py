"""OFTI as a host-loop refit boundary, with the physical model it targets.

`OftiRefit` refits one subject's relative astrometry rows with photomancy's OFTI
rejection sampler. It is a `HostLoopRefit`: the core freezes the world, hands each
dirty audience and subject to `fit` on the host, and installs the particles that
come back. The boundary is stateless, a pure function of the request rows and the
refit key, so equal rows and keys give equal particles.

`InferenceModel` declares what a refit samples (prior, likelihood, support,
parameterization and mass model). Its digest is the `model_hash` a run document
records for the boundary; two boundaries compare only within one hash.
`ofti_model` declares OFTI's target and `ofti_binding` is the `BindingKind` a run
document names the boundary by.

Statuses, per audience and subject:

- `OK`: `n_particles` orbits accepted; they are installed.
- `NO_SUPPORT`: nothing to fit (no counting rows), or OFTI ran out of batches
  before it accepted `n_particles` orbits; the belief is left as it was.
- `FAILED`: a row that cannot be fit (non-finite value, non-positive uncertainty,
  correlation of magnitude one or more) or non-finite particles; the belief is left
  as it was.
- `DEGRADED` is never returned: OFTI either accepts the full particle count or it
  does not.
"""

import dataclasses
import hashlib
import json

import jax.numpy as jnp
import numpy as np
from photomancy.orbit.data import RelativeAstromData
from photomancy.orbit.ofti import PARAM_NAMES, ofti

from spaceodyssey.codec.bindings import BindingKind, Bindings
from spaceodyssey.core.capacities import require_x64
from spaceodyssey.core.enums import RefitStatus
from spaceodyssey.core.refresh import HostLoopRefit, slice_spec

# Payload columns an astrometry row carries, in this order: epoch (days), east and
# north offsets (arcsec), their uncertainties (arcsec) and the correlation.
COLUMN_NAMES = ("time_d", "ra", "dec", "ra_err", "dec_err", "corr")

# The message OFTI raises when its batches run out; any other error is not a status.
_EXHAUSTED = "OFTI accepted"

# A hash fits a JSON number every consumer reads exactly.
_HASH_BYTES = 6


@dataclasses.dataclass(frozen=True)
class InferenceModel:
    """What a refit samples: prior, likelihood, support, parameterization, mass.

    Every field is plain JSON data, so the declaration can be read, compared and
    hashed without code. `digest` is stable across processes and changes with any
    field. Two refits of one model hash are comparable; a different hash is a
    different target.
    """

    name: str
    prior: dict
    likelihood: dict
    support: dict
    parameterization: tuple[str, ...]
    mass_model: dict

    def declaration(self):
        """The model as a plain JSON object."""
        out = dataclasses.asdict(self)
        out["parameterization"] = list(self.parameterization)
        return out

    def digest(self):
        """A positive integer from the canonical bytes of the declaration."""
        text = json.dumps(
            self.declaration(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return int.from_bytes(hashlib.sha256(text.encode()).digest()[:_HASH_BYTES])


def ofti_model(
    *,
    stellar_mass_kg,
    dist_pc,
    ecc_prior="kipping13",
    log_P_range=(2.0, 5.0),
    e_max=0.99,
):
    """The physical, fixed-mass model OFTI samples, as photomancy's `ofti` defines it.

    Isotropic orientation (uniform `cos_i`, uniform argument of periapsis, uniform
    mean anomaly at the reference epoch), eccentricity from `ecc_prior`, and a prior
    on the semimajor axis and node induced by scale-and-rotate through the
    reference-epoch measurement; no independent prior on either. The support is the
    period range (log10 days) and the eccentricity cap. The mass is fixed.
    """
    lo, hi = (float(x) for x in log_P_range)
    return InferenceModel(
        name="physical_fixed_mass",
        prior={
            "orientation": "isotropic",
            "eccentricity": str(ecc_prior),
            "semimajor_axis_and_node": "induced by scale-and-rotate",
            "mean_anomaly": "uniform at the reference epoch",
        },
        likelihood={
            "data": "relative astrometry",
            "form": "bivariate Gaussian with correlation",
            "epochs": "valid nonreference epochs by rejection",
            "reference_epoch": "conditioned through a noised target",
        },
        support={"log10_period_days": [lo, hi], "eccentricity_max": float(e_max)},
        parameterization=tuple(PARAM_NAMES),
        mass_model={
            "kind": "fixed",
            "stellar_mass_kg": float(stellar_mass_kg),
            "distance_pc": float(dist_pc),
        },
    )


def _reference_epoch(ra, dec, ra_err, dec_err, valid):
    """Index of the highest signal-to-noise valid epoch, as OFTI chooses it."""
    snr = np.hypot(ra, dec) / np.maximum(np.hypot(ra_err, dec_err), 1e-30)
    return int(np.argmax(np.where(valid, snr, -np.inf)))


def _strict_bound(ra_err, dec_err, corr, valid, ref):
    """Upper bound of the nonreference log-likelihood: the Gaussian normalization."""
    keep = valid.copy()
    keep[ref] = False
    norm = 2.0 * np.pi * ra_err * dec_err * np.sqrt(1.0 - corr**2)
    return float(-np.sum(np.log(norm[keep])))


class OftiRefit(HostLoopRefit):
    """Refit a subject's astrometry rows with photomancy's OFTI.

    Each counting row of an entry carries, in payload columns `columns` (indices, in
    `COLUMN_NAMES` order), an epoch in days, offsets and uncertainties in arcsec and
    a correlation. The observer's `subject_leaves` must be exactly the `particles`
    leaf, shaped `[n_particles, 7]` in `PARAM_NAMES` order, and optionally the
    integer `count` leaf, which receives the number of epochs fit.

    The sampler is keyed by the refit key alone. `loglike_ref` is `"strict"` (the
    sum of the log normalizations of the nonreference epochs, a true bound, so the
    particles sample the posterior) or `"first_batch"` (OFTI's empirical bound).
    Neither enters the model hash, which declares the target and not the sampler.
    """

    kind = "photomancy.ofti"

    def __init__(
        self,
        *,
        stellar_mass_kg,
        dist_pc,
        ecc_prior="kipping13",
        log_P_range=(2.0, 5.0),
        e_max=0.99,
        columns=(0, 1, 2, 3, 4, 5),
        particles="samples",
        count=None,
        batch=100_000,
        max_batches=1000,
        loglike_ref="strict",
        version=1,
    ):
        """Declare the target and the sampler budget; no study is bound."""
        if loglike_ref not in ("strict", "first_batch"):
            raise ValueError("loglike_ref is 'strict' or 'first_batch'")
        if len(columns) != len(COLUMN_NAMES):
            raise ValueError(f"columns names the payload index of {COLUMN_NAMES}")
        self.model = ofti_model(
            stellar_mass_kg=stellar_mass_kg,
            dist_pc=dist_pc,
            ecc_prior=ecc_prior,
            log_P_range=log_P_range,
            e_max=e_max,
        )
        self.stellar_mass_kg = float(stellar_mass_kg)
        self.dist_pc = float(dist_pc)
        self.ecc_prior = str(ecc_prior)
        self.log_P_range = tuple(float(x) for x in log_P_range)
        self.e_max = float(e_max)
        self.columns = tuple(int(c) for c in columns)
        self.particles = str(particles)
        self.count = None if count is None else str(count)
        self.batch = int(batch)
        self.max_batches = int(max_batches)
        self.loglike_ref = loglike_ref
        super().__init__(
            self._fit, n_diag=2, model_hash=self.model.digest(), version=version
        )

    def configuration(self):
        """The constructor arguments as plain JSON."""
        return {
            "stellar_mass_kg": self.stellar_mass_kg,
            "dist_pc": self.dist_pc,
            "ecc_prior": self.ecc_prior,
            "log_P_range": list(self.log_P_range),
            "e_max": self.e_max,
            "columns": list(self.columns),
            "particles": self.particles,
            "count": self.count,
            "batch": self.batch,
            "max_batches": self.max_batches,
            "loglike_ref": self.loglike_ref,
        }

    def _layout(self, study):
        """The particle count, checked against the observer's subject leaves."""
        spec = {name: (shape, dtype) for name, shape, dtype in slice_spec(study)}
        wanted = {self.particles} | ({self.count} if self.count else set())
        if set(spec) != wanted:
            raise ValueError(
                f"the observer's subject leaves are {sorted(spec)}; an OFTI refit "
                f"fills exactly {sorted(wanted)}"
            )
        shape = spec[self.particles][0]
        if len(shape) != 2 or shape[1] != len(PARAM_NAMES):
            raise ValueError(
                f"leaf {self.particles!r} has tail shape {shape}; "
                f"OFTI particles are [n, {len(PARAM_NAMES)}]"
            )
        return shape[0]

    def _slices(self, study, samples, n_epochs):
        """The installable slices, in the observer's `subject_leaves` order."""
        spec = {name: dtype for name, _, dtype in slice_spec(study)}
        filled = {self.particles: samples.astype(spec[self.particles])}
        if self.count:
            filled[self.count] = np.asarray(n_epochs, spec[self.count])
        return tuple(filled[name] for name in study.observer.subject_leaves)

    def _fit(self, study, entry):
        """OFTI over the entry's rows: `(slices or None, status, diagnostics)`."""
        require_x64()
        n_particles = self._layout(study)
        rows = entry.rows
        valid = np.asarray(rows.valid, bool)
        n_epochs = int(valid.sum())
        if n_epochs == 0:
            return None, RefitStatus.NO_SUPPORT, [0.0, 0.0]
        table = np.asarray(rows.payload, np.float64)[:, list(self.columns)]
        t, ra, dec, ra_err, dec_err, corr = table.T
        used = table[valid]
        sane = (
            np.isfinite(used).all()
            and (used[:, 3] > 0).all()
            and (used[:, 4] > 0).all()
            and (np.abs(used[:, 5]) < 1).all()
        )
        if not sane:
            return None, RefitStatus.FAILED, [float(n_epochs), 0.0]
        # Rows that do not count get harmless values so no division sees a zero.
        ra_err, dec_err = np.where(valid, ra_err, 1.0), np.where(valid, dec_err, 1.0)
        corr = np.where(valid, corr, 0.0)
        ref = _reference_epoch(ra, dec, ra_err, dec_err, valid)
        bound = (
            _strict_bound(ra_err, dec_err, corr, valid, ref)
            if self.loglike_ref == "strict"
            else None
        )
        data = RelativeAstromData(
            times=jnp.asarray(t),
            ra=jnp.asarray(ra),
            dec=jnp.asarray(dec),
            ra_err=jnp.asarray(ra_err),
            dec_err=jnp.asarray(dec_err),
            corr=jnp.asarray(corr),
            planet_id=jnp.zeros(valid.shape, int),
            is_valid=jnp.asarray(valid),
        )
        try:
            posterior = ofti(
                data,
                Ms=self.stellar_mass_kg,
                dist_pc=self.dist_pc,
                key=jnp.asarray(entry.key, jnp.uint32),
                n_accept=n_particles,
                ref_idx=ref,
                ecc_prior=self.ecc_prior,
                log_P_range=self.log_P_range,
                e_max=self.e_max,
                batch=self.batch,
                max_batches=self.max_batches,
                loglike_ref=bound,
            )
        except RuntimeError as err:
            if not str(err).startswith(_EXHAUSTED):
                raise
            return None, RefitStatus.NO_SUPPORT, [float(n_epochs), 0.0]
        samples = np.asarray(posterior.samples, np.float64)[:n_particles]
        if (
            samples.shape != (n_particles, len(PARAM_NAMES))
            or not np.isfinite(samples).all()
        ):
            return None, RefitStatus.FAILED, [float(n_epochs), 0.0]
        diag = [float(n_epochs), float(n_particles)]
        return self._slices(study, samples, n_epochs), RefitStatus.OK, diag


def ofti_binding():
    """The `BindingKind` a run document names an `OftiRefit` by."""
    return BindingKind(
        OftiRefit.kind,
        1,
        cls=OftiRefit,
        describe=lambda obj, registry: obj.configuration(),
        build=lambda config, registry: OftiRefit(**config),
    )


def photomancy_bindings():
    """A `Bindings` table holding the photomancy refit kinds."""
    return Bindings((ofti_binding(),))


__all__ = [
    "COLUMN_NAMES",
    "InferenceModel",
    "OftiRefit",
    "ofti_binding",
    "ofti_model",
    "photomancy_bindings",
]
