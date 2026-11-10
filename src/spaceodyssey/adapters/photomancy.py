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

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.constants import Msun2kg, Rearth2AU
from photomancy.orbit.data import RelativeAstromData
from photomancy.orbit.ofti import PARAM_NAMES, ofti
from photomancy.orbit.priors import ecc_distribution
from photomancy.orbit.updater import UpdaterStatus

from spaceodyssey.codec.bindings import BindingKind, Bindings
from spaceodyssey.core.capacities import require_x64
from spaceodyssey.core.enums import EVERYONE, F64, I64, Audience, RefitStatus
from spaceodyssey.core.records import PayloadColumn, RecordKind
from spaceodyssey.core.refresh import (
    AbstractRefitBoundary,
    HostLoopRefit,
    RefitResult,
    slice_spec,
)

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


def _eccentricity(name):
    """The numeric eccentricity distribution a prior name stands for."""
    dist = ecc_distribution(name)
    params = {k: float(getattr(dist, k)) for k in dist.arg_constraints}
    return {"name": str(name), "family": type(dist).__name__, "parameters": params}


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
            "eccentricity": _eccentricity(ecc_prior),
            "semimajor_axis_and_node": "induced by scale-and-rotate",
            "mean_anomaly": "uniform at the reference epoch",
            "phase": "time of periapsis back-computed from the scaled period",
        },
        likelihood={
            "data": "relative astrometry",
            "form": "bivariate Gaussian with correlation",
            "epochs": "valid nonreference epochs by rejection",
            "reference_epoch": "conditioned through a noised target",
            "anchor": "highest signal-to-noise valid epoch",
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

    def _readable(self, study):
        """Per audience, whether OFTI can read every column of every refit kind."""
        caps = study.caps
        out = []
        for audience in (Audience.OBSERVER, Audience.POLICY):
            seen = all(
                int(caps.payload_audience[kind][col]) & int(audience)
                for kind in caps.refit_kinds
                for col in self.columns
            )
            out.append(bool(seen))
        return np.asarray(out)

    def refit_batch(self, study, requests):
        """Fit the entries; an audience that cannot read a needed column gets none.

        A masked column reaches the request as zeros, which would fit as a real
        measurement. For such an audience every row is marked not counting, so its
        entry is `NO_SUPPORT` and installs nothing, and it never shares another
        audience's computation.
        """
        readable = self._readable(study)
        if readable.all():
            return super().refit_batch(study, requests)
        seen = jnp.asarray(readable)
        keep = seen[None, :, None]
        rows = dataclasses.replace(
            requests.rows, valid=requests.rows.valid & keep[..., None]
        )
        count = requests.dirty.shape[-1]
        own = jnp.arange(2 * count, dtype=requests.source.dtype).reshape(2, count)
        source = jnp.where(keep, requests.source, own)
        # an entry that pointed at a blind audience's computation computes itself
        source = jnp.where(seen[source // count], source, own)
        return super().refit_batch(
            study, dataclasses.replace(requests, rows=rows, source=source)
        )

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


# -- Decision summary ----------------------------------------------------------------
#
# The belief a decision study carries is not a particle cloud but the few numbers a
# policy decides from. `DecisionSummary` holds them per star; `summarize` turns one
# slot's weighted orbit cloud into its leaves, and `ReportRefit` fills them.
#
# A channel the summary reads (`study.channel_model`) answers four questions, each
# for a cloud `elements (n, 8)` in the physical order `ELEMENTS` and for one subject
# at `time_d`, days from the study epoch:
#
# - `detect_probability(subject, mode, time_d, elements) -> (n,)`
# - `block_snr(subject, block_mode, start_d, elements) -> (n,)`: the signal-to-noise
#   ratio a block started at `start_d` accumulates (a block is `block_time` long)
# - `sky_position(subject, time_d, elements) -> (n, 2)`: east and north, arcsec
# - `luminosity(subject) -> float`: solar luminosities
#
# and carries `n_modes`, `n_block_modes` and `tick_days` (days per clock tick).

ELEMENTS = ("ln_period", "ecc", "cos_i", "node", "argp", "tau", "ln_mass", "ln_lam")

N_GRID = 16
GRID_SPACING_D = 15.0
BLOCK_SNR = 10.0
POSITION_LAG_D = 60.0
ALBEDO = 0.3

# The class-1 rule of the ExoVista catalog filter, in scaled semimajor axis (AU per
# square root of solar luminosity) and radius (Earth radii).
CLASS1_A_SCALED = (0.95, 1.67)
CLASS1_R_LOWER = 0.8
CLASS1_R_UPPER = 1.4
DAYS_PER_YEAR = 365.25

# The truth catalog row: per slot, these columns in order, each in one group. Column
# `i` of slot `s` is `s * len(CATALOG_FIELDS) + i`.
CATALOG_FIELDS = (
    ("present", "existence"),
    ("ln_period", "ephemeris"),
    ("ecc", "ephemeris"),
    ("cos_i", "ephemeris"),
    ("node", "ephemeris"),
    ("argp", "ephemeris"),
    ("tau", "ephemeris"),
    ("ln_mass", "ephemeris"),
    ("class1", "class"),
    ("ln_lam", "class"),
)
CATALOG_GROUPS = ("existence", "ephemeris", "class")
CATALOG_KIND_NAME = "truth_catalog"
REPORT_KIND_NAME = "visit_report"


def catalog_kind(n_slots, audiences=None, *, name=CATALOG_KIND_NAME):
    """The truth-catalog record kind: per slot presence, orbit and class columns.

    `audiences` maps a group name (`CATALOG_GROUPS`) to the audience flags that read
    it; a group left out is readable by the scorer alone.
    """
    audiences = {} if audiences is None else dict(audiences)
    unknown = set(audiences) - set(CATALOG_GROUPS)
    if unknown:
        raise ValueError(f"unknown catalog groups {sorted(unknown)}")
    columns = tuple(
        PayloadColumn(f"{field}_{slot}", int(audiences.get(group, Audience.SCORER)))
        for slot in range(n_slots)
        for field, group in CATALOG_FIELDS
    )
    return RecordKind(name, columns)


def report_kind(n_slots, audience=EVERYONE, *, name=REPORT_KIND_NAME):
    """The visit-report record kind: one detected flag per slot."""
    return RecordKind(
        name,
        tuple(PayloadColumn(f"detected_{s}", int(audience)) for s in range(n_slots)),
    )


class DecisionSummary(eqx.Module):
    """What a policy decides from, per star: leaves `[n_subjects, ...]`.

    `S` slots, `G` future epochs spaced `GRID_SPACING_D` days from `ref_tick`, `M`
    detection modes and `B` block modes. `aux` is the observer's per-world slot (not
    per star) and carries no belief.
    """

    ref_tick: jax.Array  # I64 [n], tick of the refit
    n_obs: jax.Array  # I64 [n], visits read
    p_new: jax.Array  # [n, G, M], a visit first-detects a class-1 planet
    n_det: jax.Array  # I64 [n, S], detections of the slot
    p_det: jax.Array  # [n, S, G, M], detection probability of a detected slot
    p_block: jax.Array  # [n, S, G, B], a block started at the epoch meets its need
    scatter: jax.Array  # [n, S], predicted-position scatter, arcsec
    pos_mean: jax.Array  # [n, S, 2], predicted position, arcsec
    p_class: jax.Array  # [n, S], probability of class 1
    ln_period_q: jax.Array  # [n, S, 3], 16, 50 and 84 percent quantiles
    ess: jax.Array  # [n, S], Kish effective size
    status: jax.Array  # I64 [n, S], slot fit code (`SLOT_NEVER_FIT` before any fit)
    aux: jax.Array  # [n_an]

    @classmethod
    def zeros(cls, n_subjects, n_slots, n_grid, n_modes, n_blocks, n_an=0):
        """An all-zero summary of the given capacities."""
        n, s, g, m, b = n_subjects, n_slots, n_grid, n_modes, n_blocks
        return cls(
            ref_tick=jnp.zeros(n, I64),
            n_obs=jnp.zeros(n, I64),
            p_new=jnp.zeros((n, g, m), F64),
            n_det=jnp.zeros((n, s), I64),
            p_det=jnp.zeros((n, s, g, m), F64),
            p_block=jnp.zeros((n, s, g, b), F64),
            scatter=jnp.zeros((n, s), F64),
            pos_mean=jnp.zeros((n, s, 2), F64),
            p_class=jnp.zeros((n, s), F64),
            ln_period_q=jnp.zeros((n, s, 3), F64),
            ess=jnp.zeros((n, s), F64),
            status=jnp.full((n, s), SLOT_NEVER_FIT, I64),
            aux=jnp.zeros(n_an, F64),
        )

    @property
    def n_subjects(self):
        """Stars held."""
        return self.ref_tick.shape[0]

    def bytes_per_star(self):
        """Bytes one star holds in one audience's summary."""
        total = sum(
            x.nbytes for name, x in zip(SUMMARY_LEAVES, self._leaves(), strict=True)
        )
        return total // self.n_subjects

    def _leaves(self):
        return tuple(getattr(self, name) for name in SUMMARY_LEAVES)


# The per-star leaves, in `DecisionSummary` field order: the observer's subject leaves.
SUMMARY_LEAVES = tuple(
    f.name for f in dataclasses.fields(DecisionSummary) if f.name != "aux"
)


class SlotLeaves(eqx.Module):
    """One slot's summary leaves."""

    p_det: jax.Array  # [G, M]
    p_block: jax.Array  # [G, B]
    scatter: jax.Array
    pos_mean: jax.Array  # [2]
    p_class: jax.Array
    ln_period_q: jax.Array  # [3]
    ess: jax.Array


def _normalized(log_w):
    """Weights summing to one from log weights."""
    return jax.nn.softmax(log_w)


def kish_size(log_w):
    """Kish effective sample size of a log-weight vector."""
    w = _normalized(log_w)
    return 1.0 / jnp.sum(w * w)


def weighted_quantiles(x, log_w, probs=(0.16, 0.5, 0.84)):
    """Quantiles of `x` under the weights, by the weighted empirical distribution."""
    order = jnp.argsort(x)
    cum = jnp.cumsum(_normalized(log_w)[order])
    cut = jnp.searchsorted(cum, jnp.asarray(probs, cum.dtype))
    return x[order][jnp.clip(cut, 0, x.shape[0] - 1)]


def semimajor_axis_au(elements):
    """Semimajor axis in AU from `ln_period` (days) and `ln_mass` (solar masses)."""
    period_yr = jnp.exp(elements[..., 0]) / DAYS_PER_YEAR
    return (jnp.exp(elements[..., 6]) * period_yr**2) ** (1.0 / 3.0)


def class1_indicator(elements, luminosity, albedo=ALBEDO):
    """Whether each draw is a class-1 planet by the skyscapes catalog rule.

    The radius in Earth radii is `sqrt(exp(ln_lam) / albedo)`, `ln_lam` being
    `ln(albedo * radius**2)`; the albedo is the declared one.
    """
    scaled = semimajor_axis_au(elements) / jnp.sqrt(luminosity)
    radius = jnp.sqrt(jnp.exp(elements[..., 7]) / albedo)
    lo, hi = CLASS1_A_SCALED
    return (
        (scaled >= lo)
        & (scaled < hi)
        & (radius >= CLASS1_R_LOWER / jnp.sqrt(scaled))
        & (radius < CLASS1_R_UPPER)
    )


def _grid_times(channel, ref_tick, tick_days, n_grid, spacing_d):
    start = jnp.asarray(ref_tick, F64) * tick_days
    return start + spacing_d * jnp.arange(n_grid, dtype=F64)


def _detect_draws(elements, subject, channel, times):
    """Detection probability of every draw: `[G, M, n]`."""
    modes = jnp.arange(channel.n_modes)

    def at(mode, time):
        return channel.detect_probability(subject, mode, time, elements)

    return jax.vmap(lambda time: jax.vmap(lambda mode: at(mode, time))(modes))(times)


def summarize(
    elements,
    log_w,
    subject,
    channel,
    ref_tick,
    *,
    last_det_d=None,
    class1=None,
    tick_days=None,
    n_grid=N_GRID,
    spacing_d=GRID_SPACING_D,
    block_snr=BLOCK_SNR,
    albedo=ALBEDO,
    position_lag_d=POSITION_LAG_D,
):
    """One slot's summary leaves from a weighted orbit cloud.

    `elements (n, 8)` are in the order `ELEMENTS` and `log_w (n,)` are log weights.
    Detection and block probabilities come from the channel's own methods, so the
    realization, the likelihood and the summary share one sensitivity model.
    `last_det_d` is the last detection time in days (the refit time when absent) and
    the predicted position is read `position_lag_d` later. `class1` is a per-draw
    class indicator that replaces the catalog rule when the class is known.
    """
    tick_days = channel.tick_days if tick_days is None else tick_days
    w = _normalized(log_w)
    times = _grid_times(channel, ref_tick, tick_days, n_grid, spacing_d)
    p_det = jnp.einsum("gmn,n->gm", _detect_draws(elements, subject, channel, times), w)
    blocks = jnp.arange(channel.n_block_modes)

    def block(mode, start):
        return channel.block_snr(subject, mode, start, elements) >= block_snr

    met = jax.vmap(lambda t: jax.vmap(lambda b: block(b, t))(blocks))(times)
    p_block = jnp.einsum("gbn,n->gb", met.astype(F64), w)
    base = jnp.asarray(ref_tick, F64) * tick_days
    last = base if last_det_d is None else last_det_d
    xy = channel.sky_position(subject, last + position_lag_d, elements)
    mean = w @ xy
    scatter = jnp.sqrt(w @ jnp.sum((xy - mean) ** 2, axis=-1))
    if class1 is None:
        class1 = class1_indicator(elements, channel.luminosity(subject), albedo)
    return SlotLeaves(
        p_det=p_det,
        p_block=p_block,
        scatter=scatter,
        pos_mean=mean,
        p_class=w @ class1.astype(F64),
        ln_period_q=weighted_quantiles(elements[:, 0], log_w),
        ess=kish_size(log_w),
    )


def summarize_search(
    elements,
    log_w,
    occupancy,
    n_undetected,
    subject,
    channel,
    ref_tick,
    *,
    tick_days=None,
    n_grid=N_GRID,
    spacing_d=GRID_SPACING_D,
    albedo=ALBEDO,
):
    """`p_new [G, M]` of a belief: a visit first-detects a class-1 planet.

    The cloud is the belief of one undetected slot; `occupancy` is the expected
    number of class-1 planets still undetected around the star and `n_undetected`
    the slots that hold them, so each slot is occupied with `q = occupancy /
    n_undetected`. A visit finds one when an occupied slot is class 1 and detected:
    `1 - (1 - q p1) ** n_undetected`, `p1` the cloud's probability of both. The
    refit boundary passes the expected count after the null reports, not the prior:
    each slot's occupancy is updated by the evidence of its cloud (see
    `posterior_occupancy`), so a star whose nulls rule the cloud out is searched less.
    """
    tick_days = channel.tick_days if tick_days is None else tick_days
    w = _normalized(log_w)
    times = _grid_times(channel, ref_tick, tick_days, n_grid, spacing_d)
    cls = class1_indicator(elements, channel.luminosity(subject), albedo)
    draws = _detect_draws(elements, subject, channel, times)
    p1 = jnp.einsum("gmn,n->gm", draws * cls.astype(F64), w)
    q = occupancy / jnp.maximum(n_undetected, 1)
    return 1.0 - (1.0 - q * p1) ** n_undetected


def search_probability_truth(p_det, live):
    """`p_new [G, M]` under truth: `1 - prod(1 - p_det)` over the live slots.

    `p_det (S, G, M)` is each slot's detection probability and `live (S,)` marks the
    undetected, present, class-1 slots.
    """
    miss = jnp.where(live[:, None, None], 1.0 - p_det, 1.0)
    return 1.0 - jnp.prod(miss, axis=0)


# -- The report refit ----------------------------------------------------------------


def _catalog_elements(row, n_slots):
    """`[n_slots, 8]` elements (`ELEMENTS` order) of a catalog row, on the host."""
    width = len(CATALOG_FIELDS)
    cells = np.asarray(row)[: n_slots * width].reshape(n_slots, width)
    names = [f for f, _ in CATALOG_FIELDS]
    return np.stack([cells[:, names.index(e)] for e in ELEMENTS], axis=-1)


def _catalog_slots(row, n_slots):
    """`(present, class1, elements)` of every slot from a catalog payload row."""
    width = len(CATALOG_FIELDS)
    cells = row[: n_slots * width].reshape(n_slots, width)
    names = [f for f, _ in CATALOG_FIELDS]
    col = {name: cells[:, names.index(name)] for name in names}
    elements = jnp.stack([col[e] for e in ELEMENTS], axis=-1)
    return col["present"] > 0.5, col["class1"] > 0.5, elements


def _truth_entry(channel, entry, settings):
    """One entry's slices under the truth kernel, as a dict of leaves.

    `entry` holds the entry's `payload [R, P]`, `valid`, `kind`, `end_tick`, the
    refit `tick` and the `subject`; `settings` is the static layout.
    """
    n_slots, cat, rep = settings.n_slots, settings.catalog, settings.report
    is_cat = entry.valid & (entry.kind == cat)
    is_rep = entry.valid & (entry.kind == rep)
    row = entry.payload[jnp.argmax(is_cat)]
    present, class1, elements = _catalog_slots(row, n_slots)
    # an absent slot is evaluated on a harmless orbit and masked afterwards
    safe = jnp.where(present[:, None], elements, jnp.zeros(8, F64).at[0].set(5.0))
    seen = (
        entry.payload[:, : settings.report_cols[-1] + 1][:, list(settings.report_cols)]
        > 0.5
    ) & is_rep[:, None]
    n_det = seen.sum(axis=0).astype(I64)
    tick_days = channel.tick_days
    last_tick = jnp.max(jnp.where(seen, entry.end_tick[:, None], -1), axis=0)
    last_d = jnp.where(
        last_tick >= 0, last_tick.astype(F64) * tick_days, entry.tick * tick_days
    )
    one = jnp.zeros(1, F64)

    def slot(el, is1, last):
        return summarize(
            el[None],
            one,
            entry.subject,
            channel,
            entry.tick,
            last_det_d=last,
            class1=is1[None],
            n_grid=settings.n_grid,
            spacing_d=settings.spacing_d,
            block_snr=settings.block_snr,
            albedo=settings.albedo,
            position_lag_d=settings.position_lag_d,
        )

    leaves = jax.vmap(slot)(safe, class1, last_d)
    detected = (n_det > 0) & present
    live = ~(n_det > 0) & present & class1
    mask = lambda x: jnp.where(  # noqa: E731
        detected.reshape((-1,) + (1,) * (x.ndim - 1)), x, 0.0
    )
    keep = lambda x: jnp.where(  # noqa: E731
        present.reshape((-1,) + (1,) * (x.ndim - 1)), x, 0.0
    )
    has_cat = is_cat.any()
    status = jnp.where(
        has_cat, int(RefitStatus.OK), int(RefitStatus.NO_SUPPORT)
    ).astype(I64)
    leaves_out = {
        "ref_tick": jnp.asarray(entry.tick, I64),
        "n_obs": is_rep.sum().astype(I64),
        "p_new": search_probability_truth(leaves.p_det, live),
        "n_det": n_det,
        "p_det": mask(leaves.p_det),
        "p_block": mask(leaves.p_block),
        "scatter": keep(leaves.scatter),
        "pos_mean": keep(leaves.pos_mean),
        "p_class": keep(leaves.p_class),
        "ln_period_q": keep(leaves.ln_period_q),
        "ess": keep(leaves.ess),
    }
    # without a catalog row nothing was read: the leaves are zero, not row 0's values
    out = {k: jnp.where(has_cat, v, jnp.zeros_like(v)) for k, v in leaves_out.items()}
    out["status"] = jnp.full(n_slots, status, I64)
    return out


@dataclasses.dataclass(frozen=True)
class _Settings:
    """The static layout and constants a refit program is compiled for."""

    n_slots: int
    catalog: int
    report: int
    report_cols: tuple
    n_grid: int
    spacing_d: float
    block_snr: float
    albedo: float
    position_lag_d: float


@eqx.filter_jit
def _truth_lanes(channel, lanes, settings, names):
    """The truth kernel over a batch of entry lanes, one slice per leaf name."""
    out = jax.vmap(lambda en: _truth_entry(channel, en, settings))(lanes)
    return tuple(out[name] for name in names)


class _Lane(eqx.Module):
    """One entry of a lane batch, as the truth kernel reads it."""

    payload: jax.Array
    valid: jax.Array
    kind: jax.Array
    end_tick: jax.Array
    tick: jax.Array
    subject: jax.Array


# -- The posterior kernel -------------------------------------------------------------
#
# A slot with no detection is the first tier: a fixed cloud of prior draws inside the
# class-1 window of the star, reweighted by the nulls of the reports. A slot with one
# detection is refit by scale-and-rotate and a slot with two or more by the collapsed
# rounds, both from photomancy's physical-prior refit (the report term included). The
# work is batched by items: one item is one slot of one entry (or, for the first tier,
# one entry), items are grouped by engine and row bucket, and each group is padded to
# `lanes` and makes one jitted call whose lanes run in turn.

# The `status` leaf of a summary holds one code per slot, the same set under every
# kernel: a `RefitStatus`.
#
# - `SLOT_OK`: fit, with a cloud above the size floor (the truth kernel's only code).
# - `SLOT_DEGRADED`: fit, but the cloud stayed under the size floor after the retry
#   (photomancy's `LOW_ESS`). The leaves are installed and labelled; the slot cannot
#   support a claim.
# - `SLOT_NO_SUPPORT`: the fit found no orbit the data allow (photomancy's
#   `NO_SUPPORT`), or the truth kernel had no catalog row to read.
# - `SLOT_FAILED`: the fit returned a non-finite cloud.
# - `SLOT_NEVER_FIT`: no refit has touched the slot. It is outside the `RefitStatus`
#   codes (0 to 3), so it cannot be read as a fit outcome.
#
# A slot that is `NO_SUPPORT` or `FAILED` installs nothing: it keeps its previous
# summary and carries the code (`SummaryObserver.install`).
SLOT_NEVER_FIT = -1
SLOT_OK = int(RefitStatus.OK)
SLOT_NO_SUPPORT = int(RefitStatus.NO_SUPPORT)
SLOT_DEGRADED = int(RefitStatus.DEGRADED)
SLOT_FAILED = int(RefitStatus.FAILED)
SLOT_KEPT = (SLOT_NO_SUPPORT, SLOT_FAILED)

# Draws of the first tier's fixed prior cloud, and the factor a retry multiplies the
# proposals (or the draws) by.
FIRST_TIER_DRAWS = 4096
RETRY_FACTOR = 4
# Separation nodes of the sensitivity grid of each report row. The interpolation error
# of the report likelihood falls about 16 times from 256 to 1024 nodes, and the node
# count does not change the cost of a fit.
GRID_POINTS = 1024
# Half width added on each side of the population's range of ln(albedo radius^2), so
# a planet at the edge of the population sits inside the scale prior.
LN_LAM_PAD = 0.5
# Fractional half width of the stellar mass window around the catalog mass.
MASS_WINDOW = 0.10
# Row counts a report batch is padded to (the last is the capacity of the study).
ROW_BUCKETS = (8, 16)
# The fixed cloud of a star is drawn from this seed folded with the star.
CLOUD_SEED = 20261110
# The shrinkage rule: draws of the cloud around the catalog orbit, the fractional width
# of `a` and `Ag` after one, two and three or more detections, and the catalog groups
# the kernel reads (the orbit, and the photometric scale `ln_lam`, a column of the
# class group).
SHRINK_DRAWS = 500
SHRINK_WIDTHS = (0.20, 0.10, 0.025)
SHRINK_GROUPS = ("ephemeris", "class")
# Diagnostic columns of the posterior kernel.
POSTERIOR_DIAGNOSTICS = ("n_reports", "n_retried", "n_degraded", "n_failed")


def _photomancy_refit():
    """The report refit module of photomancy.

    Imported when a posterior first runs, never inside a trace: the module builds
    JAX constants as it loads.
    """
    from photomancy.orbit import refit

    return refit


def channel_elements(elements):
    """Physical elements `(..., 8)` of a photomancy cloud in the channel's convention.

    Photomancy holds the periastron phase `tau` (periods) and `ln_lam` as the log of
    albedo times radius squared in AU squared; the channel holds `tau` in days from
    the epoch and `ln_lam` with the radius in Earth radii.
    """
    el = jnp.asarray(elements)
    period = jnp.exp(el[..., 0])
    el = el.at[..., 5].set(el[..., 5] * period)
    return el.at[..., 7].set(el[..., 7] - 2.0 * jnp.log(Rearth2AU))


def photomancy_elements(elements):
    """The inverse of `channel_elements`: the periastron phase lies in [0, 1)."""
    el = jnp.asarray(elements)
    period = jnp.exp(el[..., 0])
    el = el.at[..., 5].set(jnp.mod(el[..., 5] / period, 1.0))
    return el.at[..., 7].set(el[..., 7] + 2.0 * jnp.log(Rearth2AU))


def population_ln_lam_range(population, pad=LN_LAM_PAD):
    """`(low, high)` of `ln(albedo * radius^2)` (AU^2) over a population, padded.

    `population` holds `Ag` and `Rp_AU` leaves and a `present` mask, as the world
    sources' populations do. The range is the scale prior of the report refit: it
    must hold every planet a world can draw, or the refit refuses a bright planet
    (the default Earth-twin window excludes planets under 0.6 or over 1.7 Earth radii
    at albedo 0.3).
    """
    present = np.asarray(population.present)
    scale = np.asarray(population.Ag) * np.asarray(population.Rp_AU) ** 2
    scale = scale[present & (scale > 0.0)]
    if scale.size == 0:
        raise ValueError("the population holds no planet to derive a scale range from")
    log = np.log(scale)
    return float(log.min() - pad), float(log.max() + pad)


def posterior_occupancy(prior, log_evidence):
    """Probability a slot holds a planet after reports: `q L / (1 - q + q L)`.

    `prior` is the slot's occupancy `q` before the reports and `log_evidence` the log
    of `L`, the likelihood of the reports (nulls) given a planet there over their
    likelihood given none. A likelihood of zero (`-inf`) gives zero; `q` is clipped
    to [0, 1].
    """
    q = jnp.clip(jnp.asarray(prior, F64), 0.0, 1.0)
    safe = jnp.where(q > 0.0, q, 0.5)
    log_num = jnp.log(safe) + log_evidence
    log_den = jnp.logaddexp(jnp.log1p(-safe), log_num)
    post = jnp.exp(log_num - log_den)
    return jnp.where((q > 0.0) & jnp.isfinite(log_den), post, 0.0)


def slot_status(photomancy_status, finite):
    """The slot code (`SLOT_*`) of a photomancy `UpdaterStatus` and a finiteness flag.

    Host-side, on NumPy values: `LOW_ESS` becomes `SLOT_DEGRADED` (the caller has
    already retried it), `NO_SUPPORT` stays so and a non-finite cloud is `SLOT_FAILED`.
    """
    raw = int(photomancy_status)
    if not finite:
        return SLOT_FAILED
    if raw == int(UpdaterStatus.NO_SUPPORT):
        return SLOT_NO_SUPPORT
    if raw == int(UpdaterStatus.LOW_ESS):
        return SLOT_DEGRADED
    return SLOT_OK


def class1_window(luminosity, mass_msun, half_width, albedo=ALBEDO):
    """The class-1 window of a star: `(ln_period_lo, hi, ln_lam_lo, hi)`.

    The period bounds follow from the scaled semimajor axis window at the largest and
    smallest mass of the stellar window, limited to the belief's 50 to 5000 day
    support; the scale bounds are the radius window of the rule at the declared
    albedo, in photomancy's units (AU squared).
    """
    root = jnp.sqrt(luminosity)
    a_lo, a_hi = CLASS1_A_SCALED[0] * root, CLASS1_A_SCALED[1] * root
    p_lo = DAYS_PER_YEAR * jnp.sqrt(a_lo**3 / (mass_msun * (1.0 + half_width)))
    p_hi = DAYS_PER_YEAR * jnp.sqrt(a_hi**3 / (mass_msun * (1.0 - half_width)))
    low = jnp.clip(jnp.log(p_lo), jnp.log(50.0), jnp.log(5000.0) - 1e-3)
    high = jnp.clip(jnp.log(p_hi), low + 1e-3, jnp.log(5000.0))
    r_lo = CLASS1_R_LOWER / jnp.sqrt(CLASS1_A_SCALED[1])
    shift = jnp.log(albedo) + 2.0 * jnp.log(Rearth2AU)
    return low, high, shift + 2.0 * jnp.log(r_lo), shift + 2.0 * jnp.log(CLASS1_R_UPPER)


@dataclasses.dataclass(frozen=True)
class _PostSettings:
    """The static layout and constants a posterior program is compiled for."""

    n_slots: int
    n_grid: int
    spacing_d: float
    block_snr: float
    albedo: float
    position_lag_d: float
    n_particles: int
    n_round: int
    n_rounds: int
    n_centers: int
    n_constants: int
    defensive: float
    ess_floor: float
    max_rounds: int
    mass_window: float
    ln_lam_range: object
    n_first: int
    occupancy: float
    grid_points: int
    lane_batch: int
    n_shrink: int
    shrink_widths: tuple


def _over_lanes(fn, lanes, batch):
    """`fn` over the lane axis: `vmap` when `batch` is 0, else `lax.map`.

    A `vmap` of the refits runs every lane through the slowest lane's loops. A plain
    `lax.map` (`batch` of 1) runs the lanes in turn, each paying only for itself, and
    keeps a per-lane `cond` a branch; with `batch_size=1` the map is a scan over a
    `vmap` and turns every `cond` into a select. A `batch` above one maps in chunks.
    """
    if batch <= 0:
        return jax.vmap(fn)(lanes)
    if batch == 1:
        return jax.lax.map(fn, lanes)
    return jax.lax.map(fn, lanes, batch_size=batch)


def _if_live(fn):
    """`fn` of a lane that runs only when `lane.live`; a dead lane returns zeros.

    The batch is padded to a fixed number of lanes with copies of a real lane; a copy
    is dead, so it costs one branch instead of a fit. Under a plain `lax.map` the
    `cond` stays a branch.
    """

    def run(lane):
        shapes = jax.eval_shape(fn, lane)
        dead = jax.tree_util.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), shapes)
        return jax.lax.cond(lane.live, fn, lambda _: dead, lane)

    return run


class _RowLane(eqx.Module):
    """One entry's report rows, compacted to the front, for the grid stage."""

    payload: jax.Array  # [R, width]
    use: jax.Array  # [R], a report row
    subject: jax.Array


class _FitLane(eqx.Module):
    """One slot of one entry (or the entry itself, for the first tier)."""

    subject: jax.Array
    tick: jax.Array
    key: jax.Array  # the entry's refit key
    slot: jax.Array  # folded into the key; the slot count for the first tier
    n_undetected: jax.Array
    epochs: jax.Array  # [R] days from the study epoch
    sep: jax.Array  # [R, P]
    dmag0: jax.Array
    limit: jax.Array
    detected: jax.Array  # [R]
    counts: jax.Array
    valid: jax.Array
    ra: jax.Array
    dec: jax.Array
    ra_err: jax.Array
    dec_err: jax.Array
    corr: jax.Array
    end_tick: jax.Array
    live: jax.Array  # False for a padding lane, which is skipped


class _ShrinkLane(_FitLane):
    """One detected slot of one entry, with the catalog orbit the rule centers on."""

    truth: jax.Array  # [8] catalog elements of the slot, in the channel's convention


def _grid_entry(channel, lane, grid_points):
    """Photomancy-ready arrays of one entry's reports, every slot.

    The channel's `report_data` reads the rows; each row's detection curve and noise
    level come from the channel's own sensitivity (`sensitivity_grid`), so the
    likelihood and the realization share one model. Slot arrays are `[S, R]`.

    The conversion carries four declared approximations, against the channel's own
    likelihood of a row at the true orbit (largest size over the fixture's detectable
    rows, in nats per row unless stated):

    - Midpoint epoch: the planet is placed at the exposure midpoint, while the channel
      integrates its motion over the exposure; 5.3e-2 in the 320,000 s mode (smaller
      in shorter modes).
    - Grid interpolation: the curve is interpolated linearly between `grid_points`
      separation nodes; at most 5e-3 at 1024 nodes (a Jupiter-sized planet at
      2.5 lambda/D), about 1e-3 at a signal-to-noise near 10.
    - Grid edges: the grid runs from 0.1 lambda/D to the end of the channel's
      response table, where the planet counts vanish; a planet outside the grid has
      zero expected counts, as in the channel.
    - Centroid: a detection's position is compared with the planet at the midpoint,
      while the channel reports the flux-weighted centroid over the exposure; the two
      differ by at most 0.07 of the position uncertainty and by 1e-3 at the median.
    """
    data = channel.report_data(lane.subject, lane.payload)
    row = data.row
    valid = data.valid & lane.use[:, None] & jnp.isfinite(row.detected)
    interval = data.interval
    science = lane.use & data.valid[:, 0]
    duration = jnp.where(science, interval.duration_s, 1.0)
    midpoint = jnp.where(science, interval.midpoint_jd, channel.epoch_jd)

    def one(mid, dur):
        return channel.sensitivity_grid(
            lane.subject, type(interval)(mid, dur), grid_points
        )

    sep, dmag0, limit = jax.vmap(one)(midpoint, duration)
    detected = (row.detected > 0.5) & valid
    pick = lambda x: jnp.where(valid, x, 0.0).T  # noqa: E731
    return {
        "epochs": jnp.where(science, midpoint - channel.epoch_jd, 0.0),
        "sep": sep,
        "dmag0": dmag0,
        "limit": limit,
        "detected": detected.T,
        "valid": valid.T,
        "counts": pick(row.counts),
        "ra": pick(row.ra),
        "dec": pick(row.dec),
        "ra_err": pick(row.ra_err),
        "dec_err": pick(row.dec_err),
        "corr": pick(row.corr),
    }


@eqx.filter_jit
def _grid_lanes(channel, lanes, st):
    """`_grid_entry` over a batch of entry lanes."""
    return jax.vmap(lambda lane: _grid_entry(channel, lane, st.grid_points))(lanes)


def photomancy_report(channel, subject, rows, slot, grid_points=GRID_POINTS):
    """Photomancy's `ReportData` of one slot from visit-report rows `[R, width]`.

    The rows are read through `channel.report_data`; each row's detection curve and
    noise level come from `channel.sensitivity_grid`, so the likelihood the refit
    weighs shares the sensitivity model of the realization.

    Declared approximations of the conversion (sizes are the largest per-row nats
    against the channel's likelihood at the true orbit, on the fixture's detectable
    rows):

    - Midpoint epoch: planet motion over the exposure is replaced by its position at
      the midpoint; 5.3e-2 nats in the 320,000 s mode.
    - Grid interpolation: linear between `grid_points` nodes; at most 5e-3 nats at
      1024 nodes.
    - Grid edges: 0.1 lambda/D to the end of the response table; beyond it the
      expected counts are zero, as in the channel.
    - Centroid: a detection's position is the midpoint position, not the flux-weighted
      centroid over the exposure; the shift is at most 0.07 position sigma.
    """
    refit = _photomancy_refit()
    rows = jnp.asarray(rows, F64)
    lane = _RowLane(payload=rows, use=jnp.ones(rows.shape[0], bool), subject=subject)
    g = _grid_entry(channel, lane, grid_points)
    return refit.ReportData(
        epochs=g["epochs"],
        sep_grid=g["sep"],
        dmag0_grid=g["dmag0"],
        limit_grid=g["limit"],
        is_detected=g["detected"][slot],
        counts=g["counts"][slot],
        is_valid=g["valid"][slot],
        snr_thresh=float(channel.law[0]),
    )


def _star_target(refit, channel, subject, st):
    """The photomancy target of a star: its distance, catalog mass and window."""
    mass = channel.catalog_mass_kg[subject] / Msun2kg
    return mass, refit.physical_prior_target(
        mass, channel.scenes.dist_pc[subject], st.mass_window, st.ln_lam_range
    )


def _report_of(refit, channel, lane, detected):
    return refit.ReportData(
        epochs=lane.epochs,
        sep_grid=lane.sep,
        dmag0_grid=lane.dmag0,
        limit_grid=lane.limit,
        is_detected=detected,
        counts=lane.counts,
        is_valid=lane.valid,
        snr_thresh=float(channel.law[0]),
    )


def _slot_key(lane, scale):
    """The entry's key folded with the slot, and with the retry factor."""
    key = jax.random.fold_in(jnp.asarray(lane.key, jnp.uint32), lane.slot)
    return jax.random.fold_in(key, scale)


def _fit_lane(channel, lane, st, engine, scale):
    """One slot with detections: refit, then its summary leaves and the status."""
    refit = _photomancy_refit()
    _, target = _star_target(refit, channel, lane.subject, st)
    report = _report_of(refit, channel, lane, lane.detected)
    astrometry = lane.detected & lane.valid & (lane.ra_err > 0) & (lane.dec_err > 0)
    records = RelativeAstromData(
        times=lane.epochs,
        ra=lane.ra,
        dec=lane.dec,
        ra_err=jnp.where(astrometry, lane.ra_err, 1.0),
        dec_err=jnp.where(astrometry, lane.dec_err, 1.0),
        corr=jnp.where(astrometry, lane.corr, 0.0),
        planet_id=jnp.zeros(lane.valid.shape, int),
        is_valid=astrometry,
    )
    n_round = st.n_round * scale
    key = _slot_key(lane, scale)
    if engine == "single":
        cloud = refit.scale_rotate_refit(
            target,
            records,
            key,
            n_particles=st.n_particles,
            n_proposals=n_round,
            ess_floor=st.ess_floor,
            report=report,
        )
    else:
        cloud = refit.collapsed_refit(
            target,
            records,
            key,
            n_particles=st.n_particles,
            n_round=n_round,
            n_rounds=st.n_rounds,
            n_centers=st.n_centers,
            defensive=st.defensive,
            n_constants=st.n_constants,
            ess_floor=st.ess_floor,
            report=report,
            max_rounds=st.max_rounds,
        )
    elements = channel_elements(cloud.elements)
    last = jnp.max(jnp.where(lane.detected & lane.valid, lane.end_tick, -1))
    last_d = jnp.where(last >= 0, last.astype(F64) * channel.tick_days, 0.0)
    leaves = summarize(
        elements,
        jnp.zeros(elements.shape[0], F64),
        lane.subject,
        channel,
        lane.tick,
        last_det_d=jnp.where(last >= 0, last_d, lane.tick * channel.tick_days),
        n_grid=st.n_grid,
        spacing_d=st.spacing_d,
        block_snr=st.block_snr,
        albedo=st.albedo,
        position_lag_d=st.position_lag_d,
    )
    finite = jnp.all(jnp.isfinite(elements))
    return (
        leaves.p_det,
        leaves.p_block,
        leaves.scatter,
        leaves.pos_mean,
        leaves.p_class,
        leaves.ln_period_q,
        cloud.ess,
        cloud.status.astype(I64),
        finite,
    )


def _first_tier_cloud(channel, lane, st, scale):
    """The first-tier cloud of an entry: `(cloud, luminosity)`.

    The cloud is the fixed class-1 prior draws of the star reweighted by the entry's
    null reports; `cloud.log_evidence` is the log of `L`, the likelihood of the
    nulls given a class-1 planet in the slot over their likelihood given none.
    """
    refit = _photomancy_refit()
    mass, _ = _star_target(refit, channel, lane.subject, st)
    luminosity = channel.luminosity(lane.subject)
    lnp_lo, lnp_hi, lam_lo, lam_hi = class1_window(
        luminosity, mass, st.mass_window, st.albedo
    )
    base = refit.physical_prior_target(
        mass, channel.scenes.dist_pc[lane.subject], st.mass_window
    )
    model = dataclasses.replace(base.model, lnP_lo=lnp_lo, lnP_hi=lnp_hi)
    target = type(base)(model, base.dist_pc, (lam_lo, lam_hi))
    n = st.n_first * scale
    cloud_key = jax.random.fold_in(jax.random.PRNGKey(CLOUD_SEED), lane.subject)
    pool = refit.prior_cloud(target, cloud_key, 2 * n)
    inside = class1_indicator(channel_elements(pool), luminosity, st.albedo)
    draws = pool[jnp.argsort(~inside, stable=True)[:n]]
    report = _report_of(refit, channel, lane, jnp.zeros_like(lane.detected))
    cloud = refit.first_tier(
        target,
        draws,
        report,
        _slot_key(lane, scale),
        n_particles=st.n_particles,
        ess_floor=st.ess_floor,
    )
    return cloud, luminosity


def _first_tier_lane(channel, lane, st, scale):
    """One entry's belief of an undetected slot: `p_new` and the slot leaves.

    Each undetected slot holds a class-1 planet with the prior occupancy `q`
    (`st.occupancy` spread over the undetected slots) before the entry's reports and
    with `q L / (1 - q + q L)` after them, `L` the null evidence of the first tier:
    every null lowers the chance that a planet is still there to find.
    """
    cloud, luminosity = _first_tier_cloud(channel, lane, st, scale)
    elements = channel_elements(cloud.elements)
    zeros = jnp.zeros(elements.shape[0], F64)
    n_dark = jnp.maximum(lane.n_undetected, 1)
    prior = st.occupancy / n_dark
    occupancy = n_dark * posterior_occupancy(prior, cloud.log_evidence)
    p_new = summarize_search(
        elements,
        zeros,
        occupancy,
        lane.n_undetected,
        lane.subject,
        channel,
        lane.tick,
        n_grid=st.n_grid,
        spacing_d=st.spacing_d,
        albedo=st.albedo,
    )
    p_class = jnp.mean(class1_indicator(elements, luminosity, st.albedo).astype(F64))
    finite = jnp.all(jnp.isfinite(elements))
    return (
        p_new,
        p_class,
        weighted_quantiles(elements[:, 0], zeros),
        cloud.ess,
        cloud.status.astype(I64),
        finite,
    )


def shrinkage_width(n_det, widths=SHRINK_WIDTHS):
    """Fractional width of the shrinkage cloud after `n_det` (at least 1) detections."""
    return jnp.asarray(widths, F64)[jnp.clip(n_det, 1, len(widths)) - 1]


def shrinkage_cloud(truth, key, n, width):
    """`n` orbits around the catalog orbit `truth (8,)`, elements in `ELEMENTS` order.

    The semimajor axis `a` and the geometric albedo `Ag` are log-normal about their
    catalog values with log standard deviation `width` (a fractional width, to second
    order), drawn independently. The stellar mass, the planet radius and every other
    element are held at the catalog values, so the period follows `a` at fixed mass
    (`ln P` moves by 1.5 `ln a`) and `ln_lam = ln(Ag Rp^2)` moves by `ln Ag`.
    """
    z = jax.random.normal(key, (n, 2), F64)
    cloud = jnp.broadcast_to(jnp.asarray(truth, F64), (n, 8))
    cloud = cloud.at[:, 0].add(1.5 * width * z[:, 0])
    return cloud.at[:, 7].add(width * z[:, 1])


def _shrink_lane(channel, lane, st, engine, scale):
    """One detected slot under the shrinkage rule: the cloud's summary leaves.

    Returns the tuple of `_fit_lane`; the cloud is never weighted, so its size is
    `st.n_shrink` and the code is always OK.
    """
    n_det = (lane.detected & lane.valid).sum()
    width = shrinkage_width(n_det, st.shrink_widths)
    elements = shrinkage_cloud(lane.truth, _slot_key(lane, scale), st.n_shrink, width)
    zeros = jnp.zeros(st.n_shrink, F64)
    last = jnp.max(jnp.where(lane.detected & lane.valid, lane.end_tick, -1))
    last_d = jnp.where(last >= 0, last.astype(F64) * channel.tick_days, 0.0)
    leaves = summarize(
        elements,
        zeros,
        lane.subject,
        channel,
        lane.tick,
        last_det_d=jnp.where(last >= 0, last_d, lane.tick * channel.tick_days),
        n_grid=st.n_grid,
        spacing_d=st.spacing_d,
        block_snr=st.block_snr,
        albedo=st.albedo,
        position_lag_d=st.position_lag_d,
    )
    return (
        leaves.p_det,
        leaves.p_block,
        leaves.scatter,
        leaves.pos_mean,
        leaves.p_class,
        leaves.ln_period_q,
        leaves.ess,
        jnp.asarray(int(UpdaterStatus.OK), I64),
        jnp.all(jnp.isfinite(elements)),
    )


@eqx.filter_jit
def _fit_lanes(channel, lanes, st, engine, scale):
    """A batch of slot fits, one output tuple per leaf."""
    fit = _shrink_lane if engine == "shrink" else _fit_lane
    return _over_lanes(
        _if_live(lambda lane: fit(channel, lane, st, engine, scale)),
        lanes,
        st.lane_batch,
    )


@eqx.filter_jit
def _first_tier_lanes(channel, lanes, st, scale):
    """A batch of first-tier entries."""
    return _over_lanes(
        _if_live(lambda lane: _first_tier_lane(channel, lane, st, scale)),
        lanes,
        st.lane_batch,
    )


class ReportRefit(AbstractRefitBoundary):
    """The refit boundary of a decision study: summaries from visit reports.

    `kernel="truth"` builds every leaf from the truth catalog row, a one-point cloud
    with no key and no particles: the oracle arm's belief. It reads the groups of
    `truth_groups` (all of `CATALOG_GROUPS`). The study's registry must show the
    catalog columns of those groups to both audiences; `caps`, when given, is
    checked at construction and the study is checked again on every call, because a
    hidden column reads as zero and would be taken for a value.

    `kernel="posterior"` fits each slot from the visit reports alone, with
    photomancy's physical-prior refit and its report term: the first tier for a slot
    with no detection (a fixed cloud of `FIRST_TIER_DRAWS` prior draws inside the
    class-1 window of the star, reweighted once per entry; `occupancy` is the
    expected number of class-1 planets per star it assumes), scale-and-rotate for one
    detection and the collapsed rounds for two or more. The target of a star takes
    its distance and catalog mass from the channel and a mass window of
    `mass_window` and a scale prior of `ln_lam_range` (the range of `ln(albedo
    radius^2)` in AU^2 the population can hold, required: see
    `population_ln_lam_range`). An audience that cannot read every measurement
    column of the report kind is answered `NO_SUPPORT` and installs nothing, so its
    summary stays what it was.

    A slot whose refit is under the size floor runs once more with `RETRY_FACTOR`
    times the proposals (a second compiled program per engine and row bucket, since
    the proposal count is a static shape); if it is still low the slot is
    `SLOT_DEGRADED`. The slot's `status` leaf holds one code set under every kernel
    (`SLOT_*`, a `RefitStatus`), photomancy's statuses mapped onto it by
    `slot_status`. A slot that is `NO_SUPPORT` or `FAILED` installs nothing: it keeps
    its previous summary and carries the code, so it cannot support a claim, while
    the slots that fit are installed. The entry is `FAILED` when every slot failed,
    `DEGRADED` when any slot failed or stayed low, else `OK`, and is counted in
    `diagnostics` (`POSTERIOR_DIAGNOSTICS`). The refit key of each slot is the
    entry's key folded with the slot. The posterior kernel reads no catalog group: an
    empty `truth_groups` is the only form it accepts, and arms that open one
    (`ephemeris` or `class`) raise `NotImplementedError`.

    `kernel="shrinkage"` is the shrinkage rule of a published study, taken as an
    external reference. A slot with no detection takes the posterior kernel's first
    tier. After a detection the slot's cloud is `SHRINK_DRAWS` orbits around its
    catalog orbit, with the semimajor axis `a` and the geometric albedo `Ag` log-normal
    at a fractional width of 0.20, 0.10 and 0.025 after one, two and three or more
    detections (`SHRINK_WIDTHS`); the planet radius, the stellar mass and every other
    element stay at the catalog values (see `shrinkage_cloud`). The cloud goes through
    `summarize` unweighted, so the leaves and the claim rule are those of every other
    kernel, and its key is the entry's key folded with the slot. The kernel reads the
    catalog groups of `SHRINK_GROUPS`: the orbit and the photometric scale `ln_lam`
    that `Ag` is centered on. It needs `occupancy` and `ln_lam_range` like the
    posterior kernel, for the first tier. A detected slot of an entry with no catalog
    row is `SLOT_NO_SUPPORT` and installs nothing.

    The first tier's `p_new` weighs each undetected slot by its posterior occupancy
    after the entry's nulls (`posterior_occupancy`), so repeated nulls on a star lower
    the chance of a new detection.

    `refit_batch` gathers the dirty entries of every world, shares duplicates through
    `requests.source`, pads to `lanes` and makes one jitted call per group of
    `lanes` items, then scatters the slices back. The posterior fits run under a
    plain `lax.map` (`lane_batch=1`): under `vmap` every lane runs the slowest lane's
    ladder loops, which made four collapsed fits far slower than the same four in
    turn. The padding lanes of a group are skipped by a `cond`, not fit. Every lane
    is independent of the others, so an entry's slices are bitwise equal at a fixed
    lane count wherever the entry sits, and agree to 1e-11 (relative) across lane
    counts (3.3e-12 measured). The stored `n_particles` is the cloud size a posterior
    kernel returns; the truth kernel ignores it and the refit key.
    """

    kind = "spaceodyssey.report_summary"

    def __init__(
        self,
        kernel="truth",
        truth_groups=CATALOG_GROUPS,
        lanes=32,
        n_particles=1024,
        *,
        catalog_kind=CATALOG_KIND_NAME,
        report_kind=REPORT_KIND_NAME,
        n_grid=N_GRID,
        spacing_d=GRID_SPACING_D,
        block_snr=BLOCK_SNR,
        albedo=ALBEDO,
        position_lag_d=POSITION_LAG_D,
        occupancy=None,
        mass_window=MASS_WINDOW,
        ln_lam_range=None,
        n_round=2**16,
        n_rounds=3,
        n_centers=256,
        n_constants=8,
        defensive=0.1,
        ess_floor=400.0,
        max_rounds=8,
        n_first=FIRST_TIER_DRAWS,
        grid_points=GRID_POINTS,
        row_buckets=ROW_BUCKETS,
        lane_batch=1,
        caps=None,
        version=1,
    ):
        """Declare the kernel and its settings; check the registry when given."""
        if kernel not in ("truth", "posterior", "shrinkage"):
            raise NotImplementedError(
                f"kernel {kernel!r} is not built; known: 'truth', 'posterior', "
                "'shrinkage'"
            )
        groups = tuple(truth_groups)
        if kernel == "truth" and set(groups) != set(CATALOG_GROUPS):
            raise ValueError(f"the truth kernel reads exactly {CATALOG_GROUPS}")
        if kernel == "shrinkage" and set(groups) != set(SHRINK_GROUPS):
            raise ValueError(f"the shrinkage kernel reads exactly {SHRINK_GROUPS}")
        if kernel in ("posterior", "shrinkage"):
            if kernel == "posterior" and groups:
                if set(groups) != set(CATALOG_GROUPS):
                    raise NotImplementedError(
                        "the posterior kernel reads no catalog group; an arm that "
                        f"opens {groups} needs the truth-informed variants"
                    )
            if occupancy is None:
                raise ValueError(
                    f"the {kernel} kernel needs `occupancy`, the class-1 planets "
                    "per star its first tier assumes"
                )
            if ln_lam_range is None:
                raise ValueError(
                    f"the {kernel} kernel needs `ln_lam_range`, the range of "
                    "ln(albedo radius^2) of the population (see "
                    "`population_ln_lam_range`); a default window would refuse a "
                    "planet outside it"
                )
            if kernel == "posterior":
                groups = ()
        if lanes < 1:
            raise ValueError("lanes is at least one")
        self.kernel = kernel
        self.truth_groups = groups
        self.lanes = int(lanes)
        self.n_particles = int(n_particles)
        self.catalog_kind = str(catalog_kind)
        self.report_kind = str(report_kind)
        self.n_grid = int(n_grid)
        self.spacing_d = float(spacing_d)
        self.block_snr = float(block_snr)
        self.albedo = float(albedo)
        self.position_lag_d = float(position_lag_d)
        self.occupancy = None if occupancy is None else float(occupancy)
        self.mass_window = float(mass_window)
        self.ln_lam_range = (
            None if ln_lam_range is None else tuple(float(x) for x in ln_lam_range)
        )
        self.n_round = int(n_round)
        self.n_rounds = int(n_rounds)
        self.n_centers = int(n_centers)
        self.n_constants = int(n_constants)
        self.defensive = float(defensive)
        self.ess_floor = float(ess_floor)
        self.max_rounds = int(max_rounds)
        self.n_first = int(n_first)
        self.grid_points = int(grid_points)
        self.row_buckets = tuple(int(b) for b in row_buckets)
        self.lane_batch = int(lane_batch)
        self.n_diag = len(POSTERIOR_DIAGNOSTICS) if kernel != "truth" else 2
        self.version = version
        text = json.dumps(self.configuration(), sort_keys=True)
        self.model_hash = int.from_bytes(
            hashlib.sha256(text.encode()).digest()[:_HASH_BYTES]
        )
        if caps is not None:
            self.check(caps)

    def configuration(self):
        """The settings that define the target, as plain JSON.

        `lanes` is left out, and so is `n_particles` under the truth kernel: neither
        changes a truth slice.
        """
        out = {
            "kernel": self.kernel,
            "truth_groups": list(self.truth_groups),
            "catalog_kind": self.catalog_kind,
            "report_kind": self.report_kind,
            "n_grid": self.n_grid,
            "spacing_d": self.spacing_d,
            "block_snr": self.block_snr,
            "albedo": self.albedo,
            "position_lag_d": self.position_lag_d,
        }
        if self.kernel != "truth":
            out |= {
                "n_particles": self.n_particles,
                "occupancy": self.occupancy,
                "mass_window": self.mass_window,
                "ln_lam_range": self.ln_lam_range,
                "n_round": self.n_round,
                "n_rounds": self.n_rounds,
                "n_centers": self.n_centers,
                "n_constants": self.n_constants,
                "defensive": self.defensive,
                "ess_floor": self.ess_floor,
                "max_rounds": self.max_rounds,
                "n_first": self.n_first,
                "grid_points": self.grid_points,
                "row_buckets": list(self.row_buckets),
                "retry_factor": RETRY_FACTOR,
                "cloud_seed": CLOUD_SEED,
            }
        if self.kernel == "shrinkage":
            out |= {
                "n_shrink": SHRINK_DRAWS,
                "shrink_widths": list(SHRINK_WIDTHS),
            }
        return out

    def check(self, caps):
        """Raise unless the registry shows the kernel every column it needs."""
        names = caps.kind_names
        if self.kernel == "posterior":
            if self.report_kind not in names:
                raise ValueError(
                    f"the registry holds {list(names)}; the refit reads "
                    f"{self.report_kind!r}"
                )
            return
        if self.catalog_kind not in names or self.report_kind not in names:
            raise ValueError(
                f"the registry holds {list(names)}; the refit reads "
                f"{self.catalog_kind!r} and {self.report_kind!r}"
            )
        row = caps.payload_audience[names.index(self.catalog_kind)]
        width = len(CATALOG_FIELDS)
        need = int(Audience.OBSERVER) | int(Audience.POLICY)
        # A declared column always has a nonempty audience; the padding to the widest
        # registered kind has none, so the slot count is the declared width.
        n_slots = sum(1 for a in row if a) // width
        for col, audience in enumerate(row[: n_slots * width]):
            field, group = CATALOG_FIELDS[col % width]
            if group in self.truth_groups and (int(audience) & need) != need:
                raise ValueError(
                    f"catalog column {field!r} of slot {col // width} (group "
                    f"{group!r}) is hidden from an audience; the {self.kernel} "
                    "kernel needs it open to the observer and the policy"
                )

    def _settings(self, study):
        spec = {name: shape for name, shape, _ in slice_spec(study)}
        names = study.caps.kind_names
        report_row = study.caps.payload_audience[names.index(self.report_kind)]
        n_slots = spec["n_det"][0]
        return _Settings(
            n_slots=n_slots,
            catalog=names.index(self.catalog_kind),
            report=names.index(self.report_kind),
            report_cols=tuple(range(n_slots)),
            n_grid=self.n_grid,
            spacing_d=self.spacing_d,
            block_snr=self.block_snr,
            albedo=self.albedo,
            position_lag_d=self.position_lag_d,
        ), len(report_row)

    def refit_batch(self, study, requests):
        """Answer every dirty entry: `[W, 2, n_subjects, ...]` slices and statuses."""
        self.check(study.caps)
        if self.kernel != "truth":
            return self._refit_posterior(study, requests)
        return self._refit_truth(study, requests)

    def _refit_truth(self, study, requests):
        req = jax.tree_util.tree_map(np.asarray, requests)
        n_worlds, _, count = req.dirty.shape
        spec = slice_spec(study)
        names = tuple(name for name, _, _ in spec)
        settings, _ = self._settings(study)
        flat = np.arange(2 * count).reshape(2, count)
        own = req.dirty & (req.source == flat[None])
        found = np.argwhere(own)
        slices = [np.zeros((n_worlds, 2, count, *s), d) for _, s, d in spec]
        status = np.full((n_worlds, 2, count), int(RefitStatus.NO_SUPPORT), np.int64)
        diag = np.zeros((n_worlds, 2, count, self.n_diag))
        tick = np.asarray(req.tick)
        tick = tick.reshape(tick.shape + (1,) * (3 - tick.ndim))
        tick = np.broadcast_to(tick, (n_worlds, 2, count))
        channel = study.channel_model
        for start in range(0, len(found), self.lanes):
            part = found[start : start + self.lanes]
            pad = np.concatenate([part, np.repeat(part[:1], self.lanes - len(part), 0)])
            w, a, s = pad.T
            lane = _Lane(
                payload=jnp.asarray(req.rows.payload[w, a, s]),
                valid=jnp.asarray(req.rows.valid[w, a, s]),
                kind=jnp.asarray(req.rows.kind[w, a, s]),
                end_tick=jnp.asarray(req.rows.end_tick[w, a, s]),
                tick=jnp.asarray(tick[w, a, s]),
                subject=jnp.asarray(s),
            )
            out = _truth_lanes(channel, lane, settings, names)
            n_part = len(part)
            wp, ap, sp = part.T
            for piece, value in zip(slices, out, strict=True):
                piece[wp, ap, sp] = np.asarray(value)[:n_part]
            code = np.asarray(out[names.index("status")])[:n_part]
            status[wp, ap, sp] = np.where(
                (code == int(RefitStatus.OK)).all(axis=1),
                int(RefitStatus.OK),
                int(RefitStatus.NO_SUPPORT),
            )
            live = np.asarray(out[names.index("ess")])[:n_part] > 0
            diag[wp, ap, sp, 0] = np.asarray(out[names.index("n_obs")])[:n_part]
            diag[wp, ap, sp, 1] = live.sum(axis=1)
        computed = own
        for w, a, s in np.argwhere(req.dirty & ~own):
            sa, ss = divmod(int(req.source[w, a, s]), count)
            for piece in slices:
                piece[w, a, s] = piece[w, sa, ss]
            status[w, a, s] = status[w, sa, ss]
            diag[w, a, s] = diag[w, sa, ss]
        return RefitResult(
            slices=tuple(jnp.asarray(p) for p in slices),
            status=jnp.asarray(status),
            diagnostics=jnp.asarray(diag),
            computed=jnp.asarray(computed),
        )

    # -- The posterior kernel on the host ----------------------------------------------

    def _readable(self, study, channel):
        """Per audience (observer, policy): every measurement column is readable."""
        names = study.caps.kind_names
        row = study.caps.payload_audience[names.index(self.report_kind)]
        width = channel.widths()[0]
        flags = (int(Audience.OBSERVER), int(Audience.POLICY))
        return tuple(all(int(x) & flag for x in row[:width]) for flag in flags)

    def _post_settings(self, channel):
        return _PostSettings(
            n_slots=channel.n_slots,
            n_grid=self.n_grid,
            spacing_d=self.spacing_d,
            block_snr=self.block_snr,
            albedo=self.albedo,
            position_lag_d=self.position_lag_d,
            n_particles=self.n_particles,
            n_round=self.n_round,
            n_rounds=self.n_rounds,
            n_centers=self.n_centers,
            n_constants=self.n_constants,
            defensive=self.defensive,
            ess_floor=self.ess_floor,
            max_rounds=self.max_rounds,
            mass_window=self.mass_window,
            ln_lam_range=self.ln_lam_range,
            n_first=self.n_first,
            occupancy=self.occupancy,
            grid_points=self.grid_points,
            lane_batch=self.lane_batch,
            n_shrink=SHRINK_DRAWS,
            shrink_widths=SHRINK_WIDTHS,
        )

    def _run_groups(self, run, items, lane_cls_fields, lane_cls=_FitLane):
        """Run `items` (a list of `_FitLane` field dicts) in padded groups of lanes.

        `run(lanes)` is a batched call returning a tuple of arrays; the result is one
        tuple of per-item numpy values for each item, in order. The padding lanes copy
        the first item and are marked dead, so they are skipped.
        """
        out = []
        for start in range(0, len(items), self.lanes):
            part = items[start : start + self.lanes]
            padded = part + [part[0]] * (self.lanes - len(part))
            lanes = lane_cls(
                **{
                    f: jnp.asarray(np.stack([it[f] for it in padded]))
                    for f in lane_cls_fields
                },
                live=jnp.arange(self.lanes) < len(part),
            )
            result = [np.asarray(x) for x in run(lanes)]
            out.extend(tuple(x[i] for x in result) for i in range(len(part)))
        return out

    def _refit_posterior(self, study, requests):
        require_x64()
        # Import outside any trace: the module builds JAX constants as it loads.
        _photomancy_refit()
        req = jax.tree_util.tree_map(np.asarray, requests)
        n_worlds, _, count = req.dirty.shape
        spec = slice_spec(study)
        names = tuple(name for name, _, _ in spec)
        channel = study.channel_model
        st = self._post_settings(channel)
        n_slots = st.n_slots
        readable = np.asarray(self._readable(study, channel))
        flat = np.arange(2 * count).reshape(2, count)
        own = req.dirty & (req.source == flat[None])
        mine = own & readable[None, :, None]
        slices = {n: np.zeros((n_worlds, 2, count, *s), d) for n, s, d in spec}
        status = np.full((n_worlds, 2, count), int(RefitStatus.NO_SUPPORT), np.int64)
        diag = np.zeros((n_worlds, 2, count, self.n_diag))
        tick = np.asarray(req.tick)
        tick = tick.reshape(tick.shape + (1,) * (3 - tick.ndim))
        tick = np.broadcast_to(tick, (n_worlds, 2, count))
        kind_index = study.caps.kind_names.index(self.report_kind)
        shrink = self.kernel == "shrinkage"
        cat_index = study.caps.kind_names.index(self.catalog_kind) if shrink else -1
        width = channel.widths()[0]
        cap = req.rows.payload.shape[3]
        buckets = sorted({b for b in self.row_buckets if b < cap} | {cap})

        # Compact each entry's report rows to the front, padded to a bucket.
        entries = []
        for w, a, s in np.argwhere(mine):
            use = req.rows.valid[w, a, s] & (req.rows.kind[w, a, s] == kind_index)
            idx = np.flatnonzero(use)
            if len(idx) == 0:
                continue
            rows = next(b for b in buckets if b >= len(idx))
            payload = np.zeros((rows, width))
            payload[: len(idx)] = req.rows.payload[w, a, s][idx, :width]
            end = np.zeros(rows, np.int64)
            end[: len(idx)] = req.rows.end_tick[w, a, s][idx]
            used = np.arange(rows) < len(idx)
            truth, has_cat = np.zeros((n_slots, 8)), False
            if shrink:
                is_cat = req.rows.valid[w, a, s] & (req.rows.kind[w, a, s] == cat_index)
                has_cat = bool(is_cat.any())
                if has_cat:
                    row = req.rows.payload[w, a, s][int(np.argmax(is_cat))]
                    truth = _catalog_elements(row, n_slots)
            entries.append(
                {
                    "at": (w, a, s),
                    "truth": truth,
                    "has_cat": has_cat,
                    "n_rep": len(idx),
                    "rows": rows,
                    "payload": payload,
                    "use": used,
                    "end": end,
                    "key": np.asarray(req.keys[w, a, s], np.uint32),
                    "tick": int(tick[w, a, s]),
                }
            )

        # Stage one: the photomancy-ready report of every entry, grouped by row bucket.
        by_rows = {}
        for e in entries:
            by_rows.setdefault(e["rows"], []).append(e)
        for group in by_rows.values():
            for start in range(0, len(group), self.lanes):
                part = group[start : start + self.lanes]
                padded = part + [part[0]] * (self.lanes - len(part))
                lanes = _RowLane(
                    payload=jnp.asarray(np.stack([e["payload"] for e in padded])),
                    use=jnp.asarray(np.stack([e["use"] for e in padded])),
                    subject=jnp.asarray([e["at"][2] for e in padded]),
                )
                result = {
                    k: np.asarray(v) for k, v in _grid_lanes(channel, lanes, st).items()
                }
                for i, e in enumerate(part):
                    e["report"] = {k: v[i] for k, v in result.items()}

        # Stage two: slot fits by engine, the first tier once per entry.
        fits, firsts = [], []
        for e in entries:
            rep = e["report"]
            astrometry = (
                rep["detected"]
                & rep["valid"]
                & (rep["ra_err"] > 0)
                & (rep["dec_err"] > 0)
            )
            e["n_det"] = (rep["detected"] & rep["valid"]).sum(axis=1)
            e["live"] = np.flatnonzero(e["n_det"] > 0)
            e["dark"] = np.flatnonzero(e["n_det"] == 0)
            e["skipped"] = []
            base = {
                "subject": np.asarray(e["at"][2]),
                "tick": np.asarray(e["tick"]),
                "key": e["key"],
                "epochs": rep["epochs"],
                "sep": rep["sep"],
                "dmag0": rep["dmag0"],
                "limit": rep["limit"],
                "end_tick": e["end"],
                "n_undetected": np.asarray(len(e["dark"])),
            }
            for slot in e["live"]:
                if shrink and not e["has_cat"]:
                    e["skipped"].append(slot)
                    continue
                item = dict(base)
                item["slot"] = np.asarray(slot)
                if shrink:
                    item["truth"] = e["truth"][slot]
                for f in (
                    "detected",
                    "valid",
                    "counts",
                    "ra",
                    "dec",
                    "ra_err",
                    "dec_err",
                    "corr",
                ):
                    item[f] = rep[f][slot]
                n_records = int(astrometry[slot].sum())
                engine = "shrink" if shrink else "single" if n_records <= 1 else "multi"
                fits.append((e, slot, engine, item))
            if len(e["dark"]):
                first = dict(base)
                first["slot"] = np.asarray(n_slots)
                # a row counts for the first tier when every undetected slot read it
                first["valid"] = rep["valid"][e["dark"]].all(axis=0)
                for f in (
                    "detected",
                    "counts",
                    "ra",
                    "dec",
                    "ra_err",
                    "dec_err",
                    "corr",
                ):
                    first[f] = np.zeros_like(rep[f][0])
                firsts.append((e, first))

        lane_cls = _ShrinkLane if shrink else _FitLane
        fit_fields = [f.name for f in dataclasses.fields(lane_cls) if f.name != "live"]

        def run_fits(selected, scale):
            results = {}
            for engine in ("shrink",) if shrink else ("single", "multi"):
                group = [(k, it) for k, (_, _, eng, it) in selected if eng == engine]
                for rows in {it["epochs"].shape[0] for _, it in group}:
                    part = [(k, it) for k, it in group if it["epochs"].shape[0] == rows]
                    out = self._run_groups(
                        lambda lanes, engine=engine: _fit_lanes(
                            channel, lanes, st, engine, scale
                        ),
                        [it for _, it in part],
                        fit_fields,
                        lane_cls,
                    )
                    results.update({k: o for (k, _), o in zip(part, out, strict=True)})
            return results

        def run_firsts(selected, scale):
            results = {}
            for rows in {it["epochs"].shape[0] for _, (_, it) in selected}:
                part = [
                    (k, it) for k, (_, it) in selected if it["epochs"].shape[0] == rows
                ]
                out = self._run_groups(
                    lambda lanes: _first_tier_lanes(channel, lanes, st, scale),
                    [it for _, it in part],
                    first_fields,
                )
                results.update({k: o for (k, _), o in zip(part, out, strict=True)})
            return results

        first_fields = [
            f.name for f in dataclasses.fields(_FitLane) if f.name != "live"
        ]
        low = int(UpdaterStatus.LOW_ESS)
        indexed_fits = list(enumerate(fits))
        fit_out = run_fits(indexed_fits, 1)
        retried_fits = [(k, fit) for k, fit in indexed_fits if fit_out[k][7] == low]
        fit_out.update(run_fits(retried_fits, RETRY_FACTOR))
        indexed_firsts = list(enumerate(firsts))
        first_out = run_firsts(indexed_firsts, 1)
        retried_firsts = [(k, f) for k, f in indexed_firsts if first_out[k][4] == low]
        first_out.update(run_firsts(retried_firsts, RETRY_FACTOR))
        retried_ids = {id(fits[k][0]) for k, _ in retried_fits} | {
            id(firsts[k][0]) for k, _ in retried_firsts
        }

        # Assemble the slices, the entry statuses and the diagnostics.
        per_entry = {}
        for k, (e, slot, _, _) in enumerate(fits):
            per_entry.setdefault(id(e), {"e": e, "fits": {}, "first": None})["fits"][
                slot
            ] = fit_out[k]
        for k, (e, _) in enumerate(firsts):
            per_entry.setdefault(id(e), {"e": e, "fits": {}, "first": None})[
                "first"
            ] = first_out[k]
        for item in per_entry.values():
            e = item["e"]
            w, a, s = e["at"]
            codes = np.zeros(n_slots, np.int64)
            units = []
            for slot, out in item["fits"].items():
                p_det, p_block, scatter, pos, p_class, q, ess, raw, finite = out
                slices["p_det"][w, a, s, slot] = p_det
                slices["p_block"][w, a, s, slot] = p_block
                slices["scatter"][w, a, s, slot] = scatter
                slices["pos_mean"][w, a, s, slot] = pos
                slices["p_class"][w, a, s, slot] = p_class
                slices["ln_period_q"][w, a, s, slot] = q
                slices["ess"][w, a, s, slot] = ess
                codes[slot] = slot_status(raw, finite)
                units.append(codes[slot])
            if item["first"] is not None:
                p_new, p_class, q, ess, raw, finite = item["first"]
                code = slot_status(raw, finite)
                slices["p_new"][w, a, s] = p_new
                for slot in e["dark"]:
                    slices["p_class"][w, a, s, slot] = p_class
                    slices["ln_period_q"][w, a, s, slot] = q
                    slices["ess"][w, a, s, slot] = ess
                    codes[slot] = code
                units.append(code)
            for slot in e["skipped"]:
                codes[slot] = SLOT_NO_SUPPORT
                units.append(SLOT_NO_SUPPORT)
            n_low = sum(c == SLOT_DEGRADED for c in units)
            n_fail = sum(c in SLOT_KEPT for c in units)
            slices["ref_tick"][w, a, s] = e["tick"]
            slices["n_obs"][w, a, s] = e["n_rep"]
            slices["n_det"][w, a, s] = e["n_det"]
            slices["status"][w, a, s] = codes
            status[w, a, s] = (
                int(RefitStatus.FAILED)
                if n_fail == len(units)
                else int(RefitStatus.DEGRADED)
                if n_fail or n_low
                else int(RefitStatus.OK)
            )
            diag[w, a, s] = [e["n_rep"], float(id(e) in retried_ids), n_low, n_fail]
        for w, a, s in np.argwhere(req.dirty & ~own):
            sa, ss = divmod(int(req.source[w, a, s]), count)
            for piece in slices.values():
                piece[w, a, s] = piece[w, sa, ss]
            status[w, a, s] = status[w, sa, ss]
            diag[w, a, s] = diag[w, sa, ss]
        return RefitResult(
            slices=tuple(jnp.asarray(slices[n]) for n in names),
            status=jnp.asarray(status),
            diagnostics=jnp.asarray(diag),
            computed=jnp.asarray(mine),
        )


__all__ = [
    "ALBEDO",
    "BLOCK_SNR",
    "CATALOG_FIELDS",
    "CATALOG_GROUPS",
    "COLUMN_NAMES",
    "ELEMENTS",
    "SHRINK_DRAWS",
    "SHRINK_GROUPS",
    "SHRINK_WIDTHS",
    "SLOT_DEGRADED",
    "SLOT_FAILED",
    "SLOT_KEPT",
    "SLOT_NEVER_FIT",
    "SLOT_NO_SUPPORT",
    "SLOT_OK",
    "SUMMARY_LEAVES",
    "DecisionSummary",
    "InferenceModel",
    "OftiRefit",
    "ReportRefit",
    "SlotLeaves",
    "catalog_kind",
    "channel_elements",
    "class1_indicator",
    "class1_window",
    "kish_size",
    "ofti_binding",
    "ofti_model",
    "photomancy_bindings",
    "photomancy_elements",
    "photomancy_report",
    "population_ln_lam_range",
    "posterior_occupancy",
    "report_kind",
    "search_probability_truth",
    "shrinkage_cloud",
    "shrinkage_width",
    "slot_status",
    "summarize",
    "summarize_search",
    "weighted_quantiles",
]
