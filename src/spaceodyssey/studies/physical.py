"""Physical study pieces for the OFTI stand-in refit: observer, refit, plan, rows.

The stand-in refits summary records (the record kind `summary_kind` declares) with
photomancy's OFTI rejection sampler. It is an approximation with a declared domain,
and every output says so (`STANDIN_SCOPE`, carried in the refit's model declaration,
its model hash and its run-document configuration):

- It is conditional on detection: only rows with ``detected = 1`` and
  ``valid_science = 1`` count as astrometry. Null rows and rows without a certified
  science interval never enter the fit.
- Its target is OFTI's induced prior: semimajor axis and node through
  scale-and-rotate onto the reference measurement, which is not the prior a
  `PhysicalWorldSource` draws truth from.
- Astrometry is instantaneous at the science-interval midpoint (the record epoch),
  not integrated over the exposure.
- Informative nulls are ignored: an entry with no detection row is `NO_SUPPORT` and
  leaves the belief unchanged, whatever its nulls say.
- Its domain is loose astrometry, where the rejection sampler accepts often enough to
  fill its particle set. The scripted rows use `STANDIN_KAPPA`, about 20 mas per axis
  at signal-to-noise 10 on the profile's response, for that reason.

Pieces:

- `ParticleObserver` holds OFTI particles per subject and changes only by refresh.
- `StandInOftiRefit` is the refit boundary: an `OftiRefit` over the summary columns,
  with record epochs read as days from the profile's time origin (photomancy's
  `tp` origin, the study epoch).
- `fixed_plan` is the fixed visit plan of the profile's first target: five
  `PlannedVisit` rows with one day of settling and one 1000 s frame each.
- `scripted_rows` builds summary payloads for a plan from a known orbit, noiseless
  or drawn through `sample_report`; `PhysicalScript` writes them as records and
  `physical_study` is the study that runs them on a refresh grid.
"""

import copy
import dataclasses
import hashlib
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.constants import Msun2kg
from hwoutils.conversions import lambda_d_to_arcsec
from orbix.equations.orbit import period_to_sma
from photomancy.orbit.forward import predict_relative_astrometry
from photomancy.orbit.ofti import PARAM_NAMES

from spaceodyssey.adapters.jaxedith import (
    CADENCE_CODES,
    CALIBRATION_ID_BYTES,
    SUMMARY_COLUMNS,
    SUMMARY_KIND_NAME,
    CountBudget,
    ReportingLaw,
    SummaryRow,
    observed_snr,
    sample_report,
    summary_kind,
)
from spaceodyssey.adapters.photomancy import (
    COLUMN_NAMES,
    InferenceModel,
    OftiRefit,
)
from spaceodyssey.adapters.physical_context import PlannedVisit
from spaceodyssey.adapters.physical_profile import load_profile
from spaceodyssey.codec.bindings import BindingKind, Bindings
from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.enums import F64, I64, Audience, RecordStatus
from spaceodyssey.core.kinds import AbstractObserver, ObserverCapabilities
from spaceodyssey.core.records import KindRegistry, RecordKind
from spaceodyssey.core.refresh import REFRESH_PRODUCT_KIND, RefreshSchedule
from spaceodyssey.core.snapshot import BeliefLayout
from spaceodyssey.studies import toy
from spaceodyssey.studies.scripted import ScriptedRecords

# The summary record's index in `physical_registry`.
SUMMARY_KIND_INDEX = 4
# Centroid error per axis in (lambda/D) / snr of the scripted rows: about 20 mas at
# signal-to-noise 10 at 1000 nm on a 7.2 m aperture, inside the stand-in's domain.
STANDIN_KAPPA = 7.0
# Planned signal-to-noise and planet-free noise [e-] of a scripted visit.
SCRIPTED_SNR = 10.0
SCRIPTED_LIMIT_COUNTS = 10.0
# What the stand-in is, as plain JSON: the model declaration and every run document
# that binds the refit carry it, readable.
STANDIN_SCOPE = {
    "approximation": "OFTI stand-in refit",
    "conditional_on": "detection: rows with detected = 1 and valid_science = 1",
    "prior": (
        "OFTI's induced prior (scale-and-rotate through the reference measurement), "
        "not the world-source prior"
    ),
    "astrometry": "instantaneous at the science-interval midpoint",
    "nulls": (
        "ignored: no detection row is NO_SUPPORT and leaves the belief unchanged"
    ),
    "domain": (
        "loose astrometry the rejection sampler accepts at a usable rate: per-axis "
        "errors near 20 mas on a 0.15 arcsec orbit (kappa 7 at signal-to-noise 10 "
        "at 1000 nm on 7.2 m); not the profile's kappa 1"
    ),
    "scripted_rows": {
        "law": {
            "threshold": ReportingLaw().threshold,
            "kappa": STANDIN_KAPPA,
            "corr": ReportingLaw().corr,
        },
        "planned_snr": SCRIPTED_SNR,
        "limit_counts": SCRIPTED_LIMIT_COUNTS,
        "position_error": "kappa (lambda/D) / snr per axis",
    },
}

# Occupancy starts of the fixed plan, days after the study epoch: visits chosen from
# the nominal orbit, outside keepout for the whole occupancy, with the nominal
# planet between the inner working angle plus 0.5 lambda/D and the field edge less
# 1 lambda/D, at least 120 days apart.
FIXED_PLAN_DAYS = (178, 298, 690, 810, 1256)
FIXED_EXPOSURE_S = 1000.0
N_PARTICLES = 16

_COLUMN = {name: i for i, name in enumerate(SUMMARY_COLUMNS)}


# -- Observer ----------------------------------------------------------------------


class ParticleBelief(eqx.Module):
    """OFTI particles per subject, the epochs each fit read, and the auxiliary slots."""

    samples: jax.Array  # [n_subjects, n_particles, 7], `PARAM_NAMES` order
    n: jax.Array  # astrometric epochs fit, per subject
    aux: jax.Array


def project_particles(belief):
    """The trace projection: the particle mean of `a` per subject, and `aux`."""
    return belief.samples[:, :, 0].mean(axis=1), belief.aux


class ParticleObserver(AbstractObserver):
    """Holds refit particles per subject; it changes only by refresh.

    `refit_kind` names the record kind it refits (the summary kind by default) and
    `n_particles` the particle count a fit fills.
    """

    subject_leaves = ("samples", "n")
    refit_kind: str = eqx.field(default=SUMMARY_KIND_NAME, static=True)
    n_particles: int = eqx.field(default=N_PARTICLES, static=True)

    def capabilities(self):
        """Refit only, from rows of `refit_kind`."""
        return ObserverCapabilities(
            incremental=False, refit=True, refit_kinds=(self.refit_kind,)
        )

    def init(self, caps):
        """Zero particles and zero epochs for every subject."""
        return ParticleBelief(
            jnp.zeros((caps.n_subjects, self.n_particles, len(PARAM_NAMES)), F64),
            jnp.zeros(caps.n_subjects, I64),
            jnp.zeros(caps.n_an, F64),
        )

    def layout(self, caps):
        """The belief is its own summary; `n` counts the epochs fit."""
        shapes = jax.eval_shape(lambda: self.init(caps))
        return BeliefLayout(
            belief=shapes, summary=shapes, count_leaf="n", project=project_particles
        )

    def install(self, belief, fitted, subject):
        """Replace one subject's particles and epoch count."""
        return ParticleBelief(
            belief.samples.at[subject].set(fitted.samples[subject]),
            belief.n.at[subject].set(fitted.n[subject]),
            belief.aux,
        )


# -- Refit -------------------------------------------------------------------------


def astrometric_rows(valid, payload):
    """Which summary rows count as astrometry: counting, detected, with science.

    `payload` holds summary columns in `SUMMARY_COLUMNS` order. A null row
    (``detected = 0``) or a row without a certified science interval
    (``valid_science = 0``) never counts, whatever its position columns hold.
    """
    payload = np.asarray(payload, np.float64)
    detected = payload[:, _COLUMN["detected"]] == 1.0
    science = payload[:, _COLUMN["valid_science"]] == 1.0
    return np.asarray(valid, bool) & detected & science


def standin_model(base, time_origin_jd):
    """`base` (an `ofti_model`) restated as the stand-in's declared target."""
    likelihood = dict(base.likelihood)
    likelihood["data"] = "summary records: east and north offsets of detection rows"
    likelihood["epochs"] = (
        "Julian Date of the science-interval midpoint, days from the time origin; "
        "rejection over the nonreference epochs"
    )
    likelihood["time_origin_jd"] = float(time_origin_jd)
    return InferenceModel(
        name="ofti_standin_conditional_on_detection",
        prior=dict(base.prior),
        likelihood=likelihood,
        support={**base.support, "scope": copy.deepcopy(STANDIN_SCOPE)},
        parameterization=base.parameterization,
        mass_model=dict(base.mass_model),
    )


class StandInOftiRefit(OftiRefit):
    """OFTI over the detection rows of summary records: an approximation.

    The scope is `STANDIN_SCOPE`: conditional on detection and on OFTI's induced
    prior, with instantaneous astrometry at the science-interval midpoint; an entry
    with no detection row is `NO_SUPPORT` and leaves the belief unchanged, so
    informative nulls are ignored. Record epochs are Julian Dates; the fit reads
    them as days from `time_origin_jd`, the origin of the particles' `tp`.

    Every keyword but `columns` is `OftiRefit`'s, and `count` defaults to the
    `ParticleObserver`'s epoch leaf `n`; the columns are the summary record's
    (`SUMMARY_COLUMNS`), and the detection and science flags are read too.
    """

    kind = "spaceodyssey.ofti_standin"

    def __init__(self, *, time_origin_jd, **kwargs):
        """Declare the target, the time origin and the sampler budget."""
        if "columns" in kwargs:
            raise TypeError("the stand-in reads the summary columns; no columns")
        columns = tuple(_COLUMN[name] for name in COLUMN_NAMES)
        kwargs.setdefault("count", "n")
        super().__init__(columns=columns, **kwargs)
        self.time_origin_jd = float(time_origin_jd)
        if not math.isfinite(self.time_origin_jd):
            raise ValueError("time_origin_jd must be finite")
        self.model = standin_model(self.model, self.time_origin_jd)
        self.model_hash = self.model.digest()

    @classmethod
    def from_profile(cls, profile=None, target=0, **kwargs):
        """The stand-in for one profile target: its host mass, distance and prior.

        The time origin is the profile's study epoch; the period support and the
        eccentricity prior are the target's declared prior.
        """
        profile = load_profile() if profile is None else profile
        entry = profile["targets"][target]
        prior = entry["planet"]["prior"]
        lo, hi = prior["period_range_d"]
        base = {
            "stellar_mass_kg": entry["mass_msun"] * Msun2kg,
            "dist_pc": entry["dist_pc"],
            "ecc_prior": prior["ecc_prior"],
            "log_P_range": (math.log10(lo), math.log10(hi)),
            "time_origin_jd": profile["time"]["study_epoch_jd"],
        }
        return cls(**{**base, **kwargs})

    def configuration(self):
        """The constructor arguments as plain JSON, with the readable scope.

        `scope` is `STANDIN_SCOPE`, so a saved run document states the
        approximation, its domain and the scripted reporting law in words, beside
        the model hash.
        """
        out = super().configuration()
        del out["columns"]
        out["time_origin_jd"] = self.time_origin_jd
        out["scope"] = copy.deepcopy(STANDIN_SCOPE)
        return out

    def _readable(self, study):
        """Per audience, whether every column the stand-in reads is readable.

        The position columns and the detection and science flags: an audience
        blind to a flag would read it as zero and count no row.
        """
        caps = study.caps
        reads = (*self.columns, _COLUMN["detected"], _COLUMN["valid_science"])
        return np.asarray(
            [
                all(
                    int(caps.payload_audience[kind][col]) & int(audience)
                    for kind in caps.refit_kinds
                    for col in reads
                )
                for audience in (Audience.OBSERVER, Audience.POLICY)
            ]
        )

    def _fit(self, study, entry):
        """OFTI over the entry's detection rows, epochs in days from the origin."""
        rows = entry.rows
        payload = np.array(rows.payload, np.float64)
        counting = astrometric_rows(rows.valid, payload)
        payload[:, _COLUMN["time_d"]] -= self.time_origin_jd
        rows = dataclasses.replace(rows, valid=counting, payload=payload)
        return super()._fit(study, dataclasses.replace(entry, rows=rows))


def _build_standin(config, registry):
    """Rebuild from a configuration; a document of another scope is refused."""
    config = dict(config)
    if config.pop("scope", None) != STANDIN_SCOPE:
        raise ValueError("the document's stand-in scope is not this stand-in's")
    return StandInOftiRefit(**config)


def standin_binding():
    """The `BindingKind` a run document names a `StandInOftiRefit` by."""
    return BindingKind(
        StandInOftiRefit.kind,
        1,
        cls=StandInOftiRefit,
        describe=lambda obj, registry: obj.configuration(),
        build=_build_standin,
    )


def standin_bindings():
    """A `Bindings` table holding the stand-in refit kind."""
    return Bindings((standin_binding(),))


# -- Plan and scripted rows --------------------------------------------------------


def fixed_plan(context, days=FIXED_PLAN_DAYS, exposure_s=FIXED_EXPOSURE_S, target=0):
    """The fixed plan: one `PlannedVisit` per occupancy start in `days`.

    Days count from tick 0 of `context.clock` (the study epoch); each visit settles
    for the context's settling time and then exposes for `exposure_s`. The plan is
    certified with `require_certified` before it is used.
    """
    clock = context.clock
    per_s = 1.0 / float(clock.clock.tick_length_s)
    exposure = round(exposure_s * per_s)
    return tuple(
        PlannedVisit(
            target=target,
            start_tick=round(day * clock.ticks_per_day),
            exposure_ticks=exposure,
        )
        for day in days
    )


@dataclasses.dataclass(frozen=True)
class ScriptedRows:
    """Scripted summary rows of a plan: occupancy ticks and payloads.

    `start_ticks` and `end_ticks` delimit each visit's occupancy (settling then
    science); `payloads` holds one summary payload per visit in `SUMMARY_COLUMNS`
    order.
    """

    start_ticks: tuple[int, ...]
    end_ticks: tuple[int, ...]
    payloads: np.ndarray


def nominal_orbit(profile=None, target=0):
    """The target's nominal orbit in OFTI's fixed-mass parameterization.

    Photomancy's names (`PARAM_NAMES`): the semimajor axis from the published
    period and the stellar mass alone, the declared inclination and node, the
    planet's argument of periastron (the star's plus 180 degrees) and the
    periastron time in days from the study epoch.
    """
    profile = load_profile() if profile is None else profile
    entry = profile["targets"][target]
    planet = entry["planet"]
    declared = planet["declared"]
    w = math.radians(planet["omega_star_deg"] + 180.0)
    return {
        "a": float(period_to_sma(planet["period_d"], entry["mass_msun"] * Msun2kg)),
        "e": planet["ecc"],
        "cos_i": math.cos(math.radians(declared["inclination_deg"])),
        "W": math.radians(declared["node_deg"]),
        "cos_w": math.cos(w),
        "sin_w": math.sin(w),
        "tp": planet["tp_jd"] - profile["time"]["study_epoch_jd"],
    }


def scripted_calibration_id(law, snr, limit_counts, lod_arcsec):
    """Six-byte digest of a scripted row's declarations; never zero."""
    text = (
        f"scripted:threshold={law.threshold!r}:kappa={law.kappa!r}:corr={law.corr!r}"
        f":snr={snr!r}:limit={limit_counts!r}:lod_arcsec={lod_arcsec!r}"
    )
    digest = hashlib.blake2b(text.encode(), digest_size=CALIBRATION_ID_BYTES).digest()
    return int.from_bytes(digest, "big") or 1


def scripted_rows(
    context,
    visits,
    orbit,
    *,
    profile=None,
    target=0,
    law=None,
    key=None,
    snr=SCRIPTED_SNR,
    limit_counts=SCRIPTED_LIMIT_COUNTS,
    valid_science=None,
):
    """Summary rows of `visits` (executed as planned) for a known `orbit`.

    Each visit occupies ``[start, start + settle + exposure]``; its epoch is the
    science-interval midpoint (`ObservingContext.science_midpoint_jd`) and its
    position is photomancy's instantaneous relative astrometry there, in OFTI's
    fixed-mass model (`orbit` in `PARAM_NAMES`, `tp` in days from the study epoch).
    The expected planet counts give signal-to-noise `snr` against the planet-free
    noise `limit_counts`.

    With `key` None the rows are noiseless detections: counts at their expected
    value, position at the truth, errors from `law` at the observed
    signal-to-noise. With a key each row is `sample_report` under `law`, keyed by
    `fold_in(key, visit)`, so a draw below threshold is a null. `valid_science`
    (one flag per visit, default all 1) marks visits without a certified science
    interval. The law defaults to `ReportingLaw(kappa=STANDIN_KAPPA)`.
    """
    profile = load_profile() if profile is None else profile
    law = ReportingLaw(kappa=STANDIN_KAPPA) if law is None else law
    entry = profile["targets"][target]
    response = profile["response"]
    lod_arcsec = float(
        lambda_d_to_arcsec(
            1.0, response["central_wavelength_nm"], response["diameter_m"]
        )
    )
    origin = profile["time"]["study_epoch_jd"]
    flags = (1,) * len(visits) if valid_science is None else tuple(valid_science)
    if len(flags) != len(visits):
        raise ValueError("valid_science needs one flag per visit")
    limit = float(limit_counts)
    planet = 0.5 * (snr**2 + snr * math.sqrt(snr**2 + 4.0 * limit**2))
    calibration = scripted_calibration_id(law, snr, limit, lod_arcsec)
    tick_s = float(context.clock.clock.tick_length_s)
    starts, ends, payloads = [], [], []
    for i, visit in enumerate(visits):
        start = int(visit.start_tick)
        end = start + int(context.settle_ticks) + int(visit.exposure_ticks)
        midpoint = context.science_midpoint_jd(start, end)
        ra, dec = predict_relative_astrometry(
            jnp.atleast_1d(midpoint - origin),
            orbit["a"],
            orbit["e"],
            orbit["cos_i"],
            orbit["W"],
            orbit["cos_w"],
            orbit["sin_w"],
            orbit["tp"],
            entry["mass_msun"] * Msun2kg,
            entry["dist_pc"],
        )
        science_s = visit.exposure_ticks * tick_s
        occupied_s = (end - start) * tick_s
        valid = bool(flags[i])
        if key is None:
            position = (ra[0], dec[0])
            row = _noiseless_row(
                law,
                planet,
                limit,
                lod_arcsec,
                midpoint,
                (science_s, occupied_s),
                position,
                calibration,
                valid,
            )
        else:
            budget = _scripted_budget(planet, limit, ra[0], dec[0], science_s)
            row = sample_report(
                jax.random.fold_in(key, i),
                law,
                budget,
                limit,
                lod_arcsec=lod_arcsec,
                time_d=midpoint,
                occupied_s=occupied_s,
                calibration_id=calibration,
                valid_science=valid,
            )
        starts.append(start)
        ends.append(end)
        payloads.append(np.asarray(row.payload(), np.float64))
    return ScriptedRows(tuple(starts), tuple(ends), np.stack(payloads))


def _scripted_budget(planet, limit, ra, dec, science_s):
    """A `CountBudget` with planet shot noise and a planet-free variance `limit^2`."""
    zero = jnp.asarray(0.0, F64)
    return CountBudget(
        planet=jnp.asarray(planet, F64),
        leakage=zero,
        zodi=zero,
        exozodi=zero,
        thermal=zero,
        var_planet=jnp.asarray(planet, F64),
        var_background=jnp.asarray(limit**2, F64),
        var_detector=zero,
        var_systematic=zero,
        ra_arcsec=jnp.asarray(ra, F64),
        dec_arcsec=jnp.asarray(dec, F64),
        science_s=jnp.asarray(science_s, F64),
        in_field=jnp.asarray(True),
        cadence=jnp.asarray(CADENCE_CODES["fixed_frame"], F64),
    )


def _noiseless_row(
    law, planet, limit, lod_arcsec, midpoint, durations, position, cal, valid
):
    """A detection row at its expected counts and true position (no science: none)."""
    science_s, occupied_s = durations
    ra, dec = position
    counts = jnp.asarray(planet if valid else 0.0, F64)
    limit = jnp.asarray(limit if valid else 0.0, F64)
    snr = observed_snr(counts, limit)
    sigma = law.position_error(snr, lod_arcsec)
    flag = jnp.asarray(1.0 if valid else 0.0, F64)
    return SummaryRow(
        time_d=jnp.asarray(midpoint, F64),
        science_s=jnp.asarray(science_s if valid else 0.0, F64),
        occupied_s=jnp.asarray(occupied_s, F64),
        detected=flag,
        counts=counts,
        limit_counts=limit,
        snr=snr,
        ra=jnp.where(valid, jnp.asarray(ra, F64), 0.0),
        dec=jnp.where(valid, jnp.asarray(dec, F64), 0.0),
        ra_err=sigma,
        dec_err=sigma,
        corr=jnp.asarray(law.corr, F64),
        cadence=jnp.asarray(CADENCE_CODES["fixed_frame"], F64),
        calibration_id=jnp.asarray(cal, F64),
        valid_science=flag,
    )


# -- Study -------------------------------------------------------------------------


def physical_registry(profile=None):
    """The toy kinds, the refresh product kind and the summary kind, in that order.

    The summary kind's columns and audiences come from `profile` (default the
    first physical profile).
    """
    return KindRegistry(
        (
            RecordKind("none", ()),
            toy.MEASUREMENT_KIND,
            toy.AUXILIARY_KIND,
            REFRESH_PRODUCT_KIND,
            summary_kind(profile),
        )
    )


class PhysicalScript(ScriptedRecords):
    """Scripted records whose payload is a whole summary row.

    Fire i happens at the end of visit i's occupancy (`ticks`) and writes a summary
    record whose start tick is the occupancy start (`starts`) and whose payload is
    `table[i]`.
    """

    starts: tuple[int, ...] = eqx.field(default=(), static=True)
    table: tuple = eqx.field(default=(), static=True)

    def handle(self, carry, ev, study):
        """Write the scripted record, then its occupancy start and summary row."""
        carry, rec = super().handle(carry, ev, study)
        row = jnp.asarray(self.table, F64)[ev.group]
        start = jnp.asarray(self.starts, I64)[ev.group]
        width = len(SUMMARY_COLUMNS)
        return carry, dataclasses.replace(
            rec, start_tick=start, payload=rec.payload.at[:width].set(row)
        )


def physical_script(rows):
    """The `PhysicalScript` that writes `rows` (a `ScriptedRows`), one per visit."""
    n = len(rows.end_ticks)
    return PhysicalScript(
        ticks=tuple(rows.end_ticks),
        values=(0.0,) * n,
        ids=tuple(range(n)),
        revisions=(0,) * n,
        statuses=(int(RecordStatus.COMPLETED),) * n,
        kinds=(SUMMARY_KIND_INDEX,) * n,
        starts=tuple(rows.start_ticks),
        table=tuple(tuple(float(x) for x in p) for p in rows.payloads),
    )


def physical_study(rows, context, *, observer=None, every_ticks=None, caps=None):
    """A study that writes `rows` and refits them on a refresh grid.

    The clock is the context's (1 ms ticks from the study epoch); the refresh grid
    is one checkpoint a day by default, so each visit's record is refit at the next
    day boundary after it. The policy is silent: the plan is executed by the script.
    """
    observer = ParticleObserver() if observer is None else observer
    caps = Capacities(rows_per_subject=8, refit_lanes=2) if caps is None else caps
    caps = physical_registry().lower(caps, observer)
    every = context.clock.ticks_per_day if every_ticks is None else every_ticks
    schedule = RefreshSchedule(every, 0, *toy.PRODUCT_CHANNELS, toy.PRODUCT_KIND)
    horizon = int(max(rows.end_ticks)) + 2 * context.clock.ticks_per_day
    study = toy.survey_1(
        caps,
        observer=observer,
        refresh=schedule,
        policies=(toy.NullPolicy(),),
        process=physical_script(rows),
        horizon_tick=horizon,
        phase_until=(horizon,),
    )
    return eqx.tree_at(
        lambda s: s.clock.tick_length_s, study, context.clock.clock.tick_length_s
    )


__all__ = [
    "FIXED_EXPOSURE_S",
    "FIXED_PLAN_DAYS",
    "N_PARTICLES",
    "SCRIPTED_LIMIT_COUNTS",
    "SCRIPTED_SNR",
    "STANDIN_KAPPA",
    "STANDIN_SCOPE",
    "SUMMARY_KIND_INDEX",
    "ParticleBelief",
    "ParticleObserver",
    "PhysicalScript",
    "ScriptedRows",
    "StandInOftiRefit",
    "astrometric_rows",
    "fixed_plan",
    "nominal_orbit",
    "physical_registry",
    "physical_script",
    "physical_study",
    "project_particles",
    "scripted_calibration_id",
    "scripted_rows",
    "standin_binding",
    "standin_bindings",
    "standin_model",
]
