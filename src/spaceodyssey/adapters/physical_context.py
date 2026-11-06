"""Absolute time, occupancy, science intervals and keepout certification.

A physical study runs on integer ticks; this module ties them to absolute time and
to where the observatory may point.

- `PhysicalClock` composes the core `Clock` with an origin: tick 0 of the study is
  `origin_ticks` ticks after J2000.0 (JD 2451545.0, TDB). Julian Dates are TDB
  float64 and MJD = JD - 2400000.5. Moving the origin one way and every tick the
  other leaves every epoch bit-identical, because the sum is formed in integers.
- `ObservingContext` holds an orbix `Observatory` (halo orbit, settling time and
  Sun keepout limits), the Earth keepout limits and the target coordinates. A
  record's start and end ticks delimit occupancy: settling, then science. The
  science interval is ``[start + settle, end]``; for a complete visit that is the
  last `exposure_ticks` of the occupancy, and its midpoint is the record epoch.
- `ObservingContext.certify` checks, before acquisition, that every admissible
  executed start in ``[planned_start, planned_start + max_delay]`` keeps the whole
  occupied interval inside the keepout window (and inside the response support,
  when a support predicate is given; it is asked to hold within half a step of
  each sample). The admissible occupied intervals cover
  ``[planned_start, planned_start + max_delay + occupancy]``, which is sampled on a
  grid of `step_d`; the keepout limits are tightened by the largest change of the
  Sun and Earth angles over half a step (`angle_rate_bound_deg_per_day` times
  ``step_d / 2``), so a sampled pass is a pass for the whole interval. The outcome
  is a named `Certification` code, never a silent refusal: no visibility at all,
  an interval straddling a keepout edge, a response-support miss, or no science
  time. `require_certified` refuses an uncertified plan at load with the findings.
- `ObservingContext.realized_science` rechecks the executed interval at run time:
  a science interval that is empty or escapes the certified window gives
  ``valid_science = 0``; the occupied time is still reported, so it is charged.
"""

import dataclasses
from enum import IntEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.constants import J2000_JD, d2s, deg2rad
from orbix.observatory import Observatory, ObservatoryL2Halo, is_observable

from spaceodyssey.core.capacities import require_x64
from spaceodyssey.core.enums import F64, I64
from spaceodyssey.core.study import Clock

# The IAU definition of the Modified Julian Date origin (hwoutils carries no MJD
# offset).
MJD_ZERO_JD = 2400000.5


class Certification(IntEnum):
    """Outcome of certifying one planned visit."""

    CERTIFIED = 0
    NO_VISIBILITY = 1  # the target is in keepout over the whole admissible interval
    KEEPOUT_EDGE = 2  # the admissible interval straddles a keepout edge
    OUTSIDE_SUPPORT = 3  # visible, but outside the response support somewhere
    NO_SCIENCE = 4  # the occupancy leaves no science time after settling


class UncertifiedPlan(ValueError):
    """A plan holds a visit that failed certification; `findings` names each one."""

    def __init__(self, findings):
        """Store the findings and list them in the message."""
        self.findings = tuple(findings)
        super().__init__("uncertified plan: " + "; ".join(self.findings))


def _ticks_per(duration_s, tick_length_s, what):
    """An exact integer number of ticks in `duration_s`, or a ValueError."""
    ticks = duration_s / tick_length_s
    whole = round(ticks)
    if whole <= 0 or abs(ticks - whole) > 1e-9 * max(1.0, abs(ticks)):
        raise ValueError(f"{what} must be a positive whole number of ticks")
    return whole


class PhysicalClock(eqx.Module):
    """The core `Clock` tied to absolute time: tick 0 is `origin_ticks` after J2000.

    Attributes:
        clock: The core clock (tick length in seconds).
        origin_ticks: Ticks from J2000.0 (JD 2451545.0, TDB) to tick 0, int64.
        ticks_per_day: Ticks in one day; a day must hold a whole number of ticks.
    """

    clock: Clock
    origin_ticks: jax.Array
    ticks_per_day: int = eqx.field(static=True)

    def __init__(self, clock, origin_ticks):
        """Store the clock and origin; refuse a tick that does not divide a day."""
        require_x64()
        self.clock = clock
        self.origin_ticks = jnp.asarray(origin_ticks, I64)
        self.ticks_per_day = _ticks_per(d2s, float(clock.tick_length_s), "a day")

    @classmethod
    def from_jd(cls, jd0, tick_length_s=1e-3):
        """A clock whose tick 0 is the TDB Julian Date `jd0`, rounded to a tick."""
        clock = Clock(tick_length_s=tick_length_s)
        per_day = _ticks_per(d2s, float(clock.tick_length_s), "a day")
        days = np.floor(jd0 - J2000_JD)
        frac = (jd0 - J2000_JD) - days
        return cls(clock, int(days) * per_day + round(frac * per_day))

    def jd(self, tick, offset_ticks=0.0):
        """TDB Julian Date of `tick` plus a fractional `offset_ticks`.

        The day count and the remainder are formed in int64 before the one
        floating-point division, so an opposite shift of `origin_ticks` and
        `tick` gives the same bits.
        """
        n = self.origin_ticks + jnp.asarray(tick, I64)
        days = jnp.floor_divide(n, self.ticks_per_day)
        rem = n - days * self.ticks_per_day
        frac = (rem.astype(F64) + jnp.asarray(offset_ticks, F64)) / self.ticks_per_day
        return (J2000_JD + days.astype(F64)) + frac

    def mjd(self, tick, offset_ticks=0.0):
        """Modified Julian Date of `tick`: exactly `jd(tick) - 2400000.5`."""
        return self.jd(tick, offset_ticks) - MJD_ZERO_JD

    def tick(self, jd):
        """The tick nearest the TDB Julian Date `jd`, int64."""
        n = jnp.round((jnp.asarray(jd, F64) - J2000_JD) * self.ticks_per_day)
        return n.astype(I64) - self.origin_ticks

    def seconds(self, ticks):
        """Duration of `ticks` in seconds."""
        return jnp.asarray(ticks, F64) * self.clock.tick_length_s


@dataclasses.dataclass(frozen=True)
class PlannedVisit:
    """One planned visit: target index, planned start, exposure and allowed delay.

    `max_delay_ticks` is the plan's slack: the executed start may fall anywhere in
    ``[start_tick, start_tick + max_delay_ticks]``.
    """

    target: int
    start_tick: int
    exposure_ticks: int
    max_delay_ticks: int = 0


class Certificate(eqx.Module):
    """A certification result: the certified window and the outcome code.

    The window is ``[window_start_tick, window_end_tick]``, the union of every
    admissible occupied interval. Only `Certification.CERTIFIED` admits science.
    """

    target: jax.Array
    window_start_tick: jax.Array
    window_end_tick: jax.Array
    outcome: jax.Array


class ScienceInterval(eqx.Module):
    """A realized visit: science interval, durations and the science flag."""

    start_tick: jax.Array
    end_tick: jax.Array
    science_ticks: jax.Array
    occupied_ticks: jax.Array
    epoch_jd: jax.Array
    valid_science: jax.Array


class ObservingContext(eqx.Module):
    """Absolute time, occupancy and keepout for a set of targets.

    Attributes:
        clock: The study's `PhysicalClock`.
        observatory: orbix `Observatory`: halo orbit, settling time and the Sun
            keepout limits (`keepout_min_deg`, `keepout_max_deg`).
        earth_min_deg: Minimum Earth angle; 0 makes the limit inactive.
        earth_max_deg: Maximum Earth angle; 180 makes the limit inactive.
        target_ra_deg: Right ascension of each target, degrees.
        target_dec_deg: Declination of each target, degrees.
        step_d: Grid step of the certification sampling, days.
        angle_rate_bound_deg_per_day: Upper bound on the rate of change of the Sun
            and Earth angles seen from the observatory; sets the keepout margin.
        settle_ticks: Settling time in ticks.
    """

    clock: PhysicalClock
    observatory: Observatory
    earth_min_deg: float
    earth_max_deg: float
    target_ra_deg: jax.Array
    target_dec_deg: jax.Array
    step_d: float
    angle_rate_bound_deg_per_day: float
    settle_ticks: int = eqx.field(static=True)

    def __init__(
        self,
        clock,
        observatory,
        target_ra_deg,
        target_dec_deg,
        *,
        earth_min_deg,
        earth_max_deg,
        step_d,
        angle_rate_bound_deg_per_day,
    ):
        """Store the fields; the settling time must be a whole number of ticks."""
        self.clock = clock
        self.observatory = observatory
        self.earth_min_deg = float(earth_min_deg)
        self.earth_max_deg = float(earth_max_deg)
        self.target_ra_deg = jnp.atleast_1d(jnp.asarray(target_ra_deg, F64))
        self.target_dec_deg = jnp.atleast_1d(jnp.asarray(target_dec_deg, F64))
        self.step_d = float(step_d)
        self.angle_rate_bound_deg_per_day = float(angle_rate_bound_deg_per_day)
        self.settle_ticks = round(
            float(observatory.settling_time_d) * clock.ticks_per_day
        )

    def __check_init__(self):
        """Refuse mismatched coordinates, a bad grid step and bad keepout limits."""
        if self.target_ra_deg.shape != self.target_dec_deg.shape:
            raise ValueError("target_ra_deg and target_dec_deg need one entry each")
        if not self.step_d > 0.0 or not self.angle_rate_bound_deg_per_day > 0.0:
            raise ValueError("step_d and angle_rate_bound_deg_per_day must be positive")
        obs = self.observatory
        for lo, hi, what in (
            (obs.keepout_min_deg, obs.keepout_max_deg, "Sun"),
            (self.earth_min_deg, self.earth_max_deg, "Earth"),
        ):
            if not 0.0 <= lo < hi <= 180.0:
                raise ValueError(
                    f"{what} keepout limits must satisfy 0 <= min < max <= 180"
                )
        if self.settle_ticks < 0:
            raise ValueError("settling time must be nonnegative")

    @property
    def n_targets(self):
        """Number of targets."""
        return int(self.target_ra_deg.shape[0])

    def occupancy_ticks(self, exposure_ticks):
        """Occupancy of a complete visit: settling plus science."""
        return self.settle_ticks + jnp.asarray(exposure_ticks, I64)

    def observable(self, target, jd, margin_deg=0.0):
        """Whether `target` is outside keepout at each TDB Julian Date in `jd`.

        Active limits (a minimum above 0 or a maximum below 180 degrees) are
        tightened by `margin_deg`; inactive ones stay inactive. The epochs are
        padded to a power of two before the compiled evaluation, so grids of
        similar length share one compilation.
        """
        mjd = jnp.atleast_1d(jnp.asarray(jd, F64)) - MJD_ZERO_JD
        n = int(mjd.shape[0])
        if n == 0:
            return jnp.zeros(0, bool)
        size = 1 << max(0, (n - 1).bit_length())
        padded = jnp.concatenate([mjd, jnp.full(size - n, mjd[-1])])
        margin = jnp.asarray(margin_deg, F64)
        return _observable(self, jnp.asarray(target, I64), padded, margin)[:n]

    def science_interval(self, start_tick, end_tick):
        """Science interval ``[start + settle, end]`` of an executed occupancy.

        Returns ``(science_start, science_end, has_science)``; with no science
        time (a visit ended during settling) the interval is empty and
        `has_science` is false.
        """
        start = jnp.asarray(start_tick, I64)
        end = jnp.asarray(end_tick, I64)
        science_start = start + self.settle_ticks
        science_end = jnp.maximum(end, science_start)
        return science_start, science_end, science_end > science_start

    def science_midpoint_jd(self, start_tick, end_tick):
        """TDB Julian Date of the science-interval midpoint, the record epoch."""
        s0, s1, _ = self.science_interval(start_tick, end_tick)
        return self.clock.jd(s0, (s1 - s0).astype(F64) / 2.0)

    def certify(self, visit, support=None):
        """Certify one `PlannedVisit`; returns a `Certificate`.

        `support`, when given, is a callable
        ``support(target, jd, margin_d=...) -> bool array`` for the response
        support (for example `SeparationSupport`, the nominal planet separation
        inside the working angles). It is evaluated on the same grid with
        ``margin_d = step_d / 2`` and must answer whether the support holds for
        every epoch within `margin_d` days of each sample, so a sampled pass
        covers the whole interval.
        """
        if not 0 <= visit.target < self.n_targets:
            raise ValueError(f"visit target {visit.target} is not a known target")
        if visit.exposure_ticks < 0 or visit.max_delay_ticks < 0:
            raise ValueError("exposure_ticks and max_delay_ticks must be nonnegative")
        occupancy = self.settle_ticks + int(visit.exposure_ticks)
        start = int(visit.start_tick)
        end = start + int(visit.max_delay_ticks) + occupancy

        def certificate(outcome):
            return Certificate(
                target=jnp.asarray(visit.target, I64),
                window_start_tick=jnp.asarray(start, I64),
                window_end_tick=jnp.asarray(end, I64),
                outcome=jnp.asarray(int(outcome), I64),
            )

        if visit.exposure_ticks == 0:
            return certificate(Certification.NO_SCIENCE)
        step = max(1, round(self.step_d * self.clock.ticks_per_day))
        ticks = np.unique(np.append(np.arange(start, end, step, dtype=np.int64), end))
        jd = self.clock.jd(ticks)
        margin = self.angle_rate_bound_deg_per_day * self.step_d / 2.0
        if not bool(jnp.all(self.observable(visit.target, jd, margin))):
            if not bool(jnp.any(self.observable(visit.target, jd))):
                return certificate(Certification.NO_VISIBILITY)
            return certificate(Certification.KEEPOUT_EDGE)
        if support is not None and not bool(
            jnp.all(support(visit.target, jd, margin_d=self.step_d / 2.0))
        ):
            return certificate(Certification.OUTSIDE_SUPPORT)
        return certificate(Certification.CERTIFIED)

    def realized_science(self, certificate, start_tick, end_tick):
        """The executed visit as a `ScienceInterval`, rechecked against `certificate`.

        `valid_science` is 1 only when the certificate is `CERTIFIED`, the science
        interval is nonempty and the whole occupancy lies inside the certified
        window. Otherwise it is 0 and the science duration is reported as 0; the
        occupied time is reported either way.
        """
        start = jnp.asarray(start_tick, I64)
        end = jnp.asarray(end_tick, I64)
        s0, s1, has_science = self.science_interval(start, end)
        inside = (start >= certificate.window_start_tick) & (
            end <= certificate.window_end_tick
        )
        valid = (
            has_science & inside & (certificate.outcome == int(Certification.CERTIFIED))
        )
        return ScienceInterval(
            start_tick=s0,
            end_tick=s1,
            science_ticks=jnp.where(valid, s1 - s0, 0),
            occupied_ticks=jnp.maximum(end - start, 0),
            epoch_jd=self.clock.jd(s0, (s1 - s0).astype(F64) / 2.0),
            valid_science=valid.astype(I64),
        )


@eqx.filter_jit
def _observable(context, target, mjd, margin_deg):
    """Keepout check of `target` at each MJD; the limits are tightened when active."""
    obs = context.observatory

    def tighten(lo, hi):
        lo = lo + margin_deg if lo > 0.0 else lo
        hi = hi - margin_deg if hi < 180.0 else hi
        return lo, hi

    sun_lo, sun_hi = tighten(obs.keepout_min_deg, obs.keepout_max_deg)
    earth_lo, earth_hi = tighten(context.earth_min_deg, context.earth_max_deg)
    ra = context.target_ra_deg[target] * deg2rad
    dec = context.target_dec_deg[target] * deg2rad

    def one(m):
        return is_observable(
            obs.orbit.position_ecliptic(m),
            ra,
            dec,
            m,
            ko_sun_min_deg=sun_lo,
            ko_sun_max_deg=sun_hi,
            ko_earth_min_deg=earth_lo,
            ko_earth_max_deg=earth_hi,
        )

    return jax.vmap(one)(mjd)


def certify_plan(context, visits, support=None):
    """Certify every visit of a plan; returns the certificates in plan order."""
    return tuple(context.certify(v, support) for v in visits)


def require_certified(context, visits, support=None):
    """The plan's certificates, or `UncertifiedPlan` naming every failed visit."""
    certificates = certify_plan(context, visits, support)
    findings = [
        f"visit {i} (target {v.target}, start tick {v.start_tick}): "
        f"{Certification(int(c.outcome)).name}"
        for i, (v, c) in enumerate(zip(visits, certificates, strict=True))
        if int(c.outcome) != Certification.CERTIFIED
    ]
    if findings:
        raise UncertifiedPlan(findings)
    return certificates


def context_from_profile(profile, clock=None, targets=None):
    """An `ObservingContext` for the profile's targets, limits and study epoch.

    `clock` defaults to a 1 ms clock whose tick 0 is the profile's study epoch.
    `targets` selects target entries by index (default all, in profile order).
    """
    time = profile["time"]
    keepout = profile["keepout"]
    if clock is None:
        clock = PhysicalClock.from_jd(time["study_epoch_jd"], time["tick_length_s"])
    entries = profile["targets"]
    chosen = range(len(entries)) if targets is None else targets
    observatory = Observatory(
        orbit=ObservatoryL2Halo.from_default(),
        settling_time_d=keepout["settling_time_d"],
        keepout_min_deg=keepout["sun_min_deg"],
        keepout_max_deg=keepout["sun_max_deg"],
    )
    return ObservingContext(
        clock,
        observatory,
        [entries[i]["ra_deg"] for i in chosen],
        [entries[i]["dec_deg"] for i in chosen],
        earth_min_deg=keepout["earth_min_deg"],
        earth_max_deg=keepout["earth_max_deg"],
        step_d=keepout["certification_step_d"],
        angle_rate_bound_deg_per_day=keepout["angle_rate_bound_deg_per_day"],
    )
