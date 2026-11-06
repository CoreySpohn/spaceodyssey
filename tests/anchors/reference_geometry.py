"""Independent geometry references: 30-digit Kepler solutions and hand formulas.

This module imports none of the libraries under test (no orbix, skyscapes,
photomancy, hwoutils or spaceodyssey). Constants come from astropy.constants and
arithmetic from mpmath at 30 significant digits.

Frame: right-handed (east, north, away) with the Thiele-Innes constants of
Green (1985, Spherical Astronomy, ch. 17): north = A X + F Y, east = B X + G Y,
away = C X + H Y, with X = cos E - e and Y = sqrt(1 - e^2) sin E in units of
the semimajor axis. The node is measured from north through east and the
ascending node is the crossing where the planet recedes (away increasing).
"""

import astropy.constants as const
import mpmath as mp

mp.mp.dps = 30

DAY_S = 86400  # seconds in a day, by definition of the Julian day
G_SI = mp.mpf(const.G.si.value)
AU_M = mp.mpf(const.au.si.value)
M_SUN_KG = mp.mpf(const.M_sun.si.value)
M_JUP_KG = mp.mpf(const.M_jup.si.value)
M_EARTH_KG = mp.mpf(const.M_earth.si.value)
R_JUP_M = mp.mpf(const.R_jup.si.value)


def kepler_eccentric_anomaly(mean_anomaly, ecc):
    """E solving E - e sin E = M to 30 digits (Newton from a safe start)."""
    m = mp.mpf(mean_anomaly) % (2 * mp.pi)
    e = mp.mpf(ecc)
    E = m if e < mp.mpf("0.8") else mp.pi
    for _ in range(100):
        step = (E - e * mp.sin(E) - m) / (1 - e * mp.cos(E))
        E -= step
        if abs(step) < mp.mpf(10) ** -28:
            break
    return E


def semimajor_axis_au(period_d, central_mass_kg):
    """Kepler's third law: a = (G M P^2 / (4 pi^2))^(1/3), in AU."""
    p_s = mp.mpf(period_d) * DAY_S
    a_m = mp.cbrt(G_SI * mp.mpf(central_mass_kg) * p_s**2 / (4 * mp.pi**2))
    return a_m / AU_M


def relative_position(
    t_jd, *, period_d, ecc, inc_deg, node_deg, omega_deg, tp_jd, a_au
):
    """Planet position relative to the star, (east, north, away) in AU."""
    n = 2 * mp.pi / mp.mpf(period_d)
    M = n * (mp.mpf(t_jd) - mp.mpf(tp_jd))
    e = mp.mpf(ecc)
    E = kepler_eccentric_anomaly(M, e)
    X = mp.cos(E) - e
    Y = mp.sqrt(1 - e**2) * mp.sin(E)
    i, W, w = (mp.radians(mp.mpf(x)) for x in (inc_deg, node_deg, omega_deg))
    a = mp.mpf(a_au)
    A = a * (mp.cos(w) * mp.cos(W) - mp.sin(w) * mp.sin(W) * mp.cos(i))
    B = a * (mp.cos(w) * mp.sin(W) + mp.sin(w) * mp.cos(W) * mp.cos(i))
    F = a * (-mp.sin(w) * mp.cos(W) - mp.cos(w) * mp.sin(W) * mp.cos(i))
    G = a * (-mp.sin(w) * mp.sin(W) + mp.cos(w) * mp.cos(W) * mp.cos(i))
    C = a * mp.sin(w) * mp.sin(i)
    H = a * mp.cos(w) * mp.sin(i)
    return B * X + G * Y, A * X + F * Y, C * X + H * Y


def time_of_true_anomaly(f_deg, *, period_d, ecc, tp_jd):
    """The epoch (JD) after `tp_jd` at which the true anomaly is `f_deg`."""
    e = mp.mpf(ecc)
    f = mp.radians(mp.mpf(f_deg)) % (2 * mp.pi)
    E = 2 * mp.atan2(mp.sqrt(1 - e) * mp.sin(f / 2), mp.sqrt(1 + e) * mp.cos(f / 2))
    M = (E - e * mp.sin(E)) % (2 * mp.pi)
    return mp.mpf(tp_jd) + M / (2 * mp.pi) * mp.mpf(period_d)


def phase_angle(east, north, away):
    """Standard phase angle atan2(rho, away): 0 at superior conjunction."""
    return mp.atan2(mp.sqrt(east**2 + north**2), away)


def lambert_phase(beta):
    """Lambert phase function (sin b + (pi - b) cos b) / pi."""
    return (mp.sin(beta) + (mp.pi - beta) * mp.cos(beta)) / mp.pi


def contrast(east, north, away, *, radius_m, albedo):
    """Grey Lambertian contrast Ag Phi(beta) (R / r)^2."""
    r_m = mp.sqrt(east**2 + north**2 + away**2) * AU_M
    beta = phase_angle(east, north, away)
    return mp.mpf(albedo) * lambert_phase(beta) * (mp.mpf(radius_m) / r_m) ** 2


def arcsec(length_au, dist_pc):
    """Angle in arcsec subtended by `length_au` at `dist_pc` (1 AU at 1 pc)."""
    return mp.mpf(length_au) / mp.mpf(dist_pc)
