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
from photomancy.orbit.data import RelativeAstromData
from photomancy.orbit.ofti import PARAM_NAMES, ofti
from photomancy.orbit.priors import ecc_distribution

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
    status: jax.Array  # I64 [n, S], refit status
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
            status=jnp.full((n, s), int(RefitStatus.NO_SUPPORT), I64),
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
    `1 - (1 - q p1) ** n_undetected`, `p1` the cloud's probability of both.
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


class ReportRefit(AbstractRefitBoundary):
    """The refit boundary of a decision study: summaries from visit reports.

    `kernel="truth"` builds every leaf from the truth catalog row, a one-point cloud
    with no key and no particles: the oracle arm's belief. It reads the groups of
    `truth_groups` (all of `CATALOG_GROUPS`). The study's registry must show the
    catalog columns of those groups to both audiences; `caps`, when given, is
    checked at construction and the study is checked again on every call, because a
    hidden column reads as zero and would be taken for a value.

    `refit_batch` gathers the dirty entries of every world, shares duplicates through
    `requests.source`, pads to `lanes` and makes one jitted call per group of
    `lanes` entries, then scatters the slices back. Every lane is independent of the
    others, so an entry's slices are the same alone or in any batch. The stored
    `n_particles` is the cloud size a posterior kernel draws; the truth kernel
    ignores it and the refit key.
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
        caps=None,
        version=1,
    ):
        """Declare the kernel and its settings; check the registry when given."""
        if kernel != "truth":
            raise NotImplementedError(f"kernel {kernel!r} is not built; only 'truth'")
        groups = tuple(truth_groups)
        if set(groups) != set(CATALOG_GROUPS):
            raise ValueError(f"the truth kernel reads exactly {CATALOG_GROUPS}")
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
        self.n_diag = 2
        self.version = version
        text = json.dumps(self.configuration(), sort_keys=True)
        self.model_hash = int.from_bytes(
            hashlib.sha256(text.encode()).digest()[:_HASH_BYTES]
        )
        if caps is not None:
            self.check(caps)

    def configuration(self):
        """The settings that define the target, as plain JSON.

        `lanes` and `n_particles` are left out: neither changes a truth slice.
        """
        return {
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

    def check(self, caps):
        """Raise unless both audiences read every catalog column the kernel needs."""
        names = caps.kind_names
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
                    f"{group!r}) is hidden from an audience; the truth kernel "
                    "needs it open to the observer and the policy"
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


__all__ = [
    "ALBEDO",
    "BLOCK_SNR",
    "CATALOG_FIELDS",
    "CATALOG_GROUPS",
    "COLUMN_NAMES",
    "ELEMENTS",
    "SUMMARY_LEAVES",
    "DecisionSummary",
    "InferenceModel",
    "OftiRefit",
    "ReportRefit",
    "SlotLeaves",
    "catalog_kind",
    "class1_indicator",
    "kish_size",
    "ofti_binding",
    "ofti_model",
    "photomancy_bindings",
    "report_kind",
    "search_probability_truth",
    "summarize",
    "summarize_search",
    "weighted_quantiles",
]
