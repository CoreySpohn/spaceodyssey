"""Named physical planet parameters and the packed truth catalog."""

import equinox as eqx
import jax
import jax.numpy as jnp
from hwoutils.constants import Mearth2kg, Msun2kg, Rearth2AU
from orbix.equations.orbit import period_to_sma
from skyscapes.population import mass_to_radius

from spaceodyssey.core.enums import EVERYONE, F64, Audience
from spaceodyssey.core.records import RecordKind


class PlanetParameters(eqx.Module):
    """Physical parameters with common leading draw axes and albedo [..., B]."""

    ln_period_d: jax.Array
    ecc: jax.Array
    cos_i: jax.Array
    node_rad: jax.Array
    argp_rad: jax.Array
    tau_d: jax.Array
    star_mass_msun: jax.Array
    planet_mass_mearth: jax.Array
    radius_rearth: jax.Array
    albedo: jax.Array


def planet_mass_mearth(radius_rearth):
    """Invert the first two Chen and Kipping (2017) branches, capped at 131.6."""
    radius = jnp.minimum(jnp.asarray(radius_rearth), mass_to_radius(131.6))
    r_break = mass_to_radius(2.04)
    mass = jnp.where(
        radius <= r_break,
        (radius / 1.008) ** (1.0 / 0.279),
        2.04 * (radius / r_break) ** (1.0 / 0.589),
    )
    return jnp.minimum(mass, 131.6)


def semimajor_axis_au(p):
    """Semimajor axis from the period and the sum of the two physical masses."""
    mass_kg = p.star_mass_msun * Msun2kg + p.planet_mass_mearth * Mearth2kg
    return period_to_sma(jnp.exp(p.ln_period_d), mass_kg)


def scale_au2(p, band_index):
    """Reflected-light scale in AU squared at a declared band index."""
    return p.albedo[..., band_index] * (p.radius_rearth * Rearth2AU) ** 2


def from_photomancy(elements, albedo, band_index):
    """Convert photomancy [n, 8] rows and albedo [n, B] to physical parameters."""
    radius = jnp.sqrt(jnp.exp(elements[..., 7]) / albedo[..., band_index]) / Rearth2AU
    mass = planet_mass_mearth(radius)
    return PlanetParameters(
        ln_period_d=elements[..., 0],
        ecc=elements[..., 1],
        cos_i=elements[..., 2],
        node_rad=elements[..., 3],
        argp_rad=elements[..., 4],
        tau_d=elements[..., 5] * jnp.exp(elements[..., 0]),
        star_mass_msun=jnp.exp(elements[..., 6]) - mass * Mearth2kg / Msun2kg,
        planet_mass_mearth=mass,
        radius_rearth=radius,
        albedo=albedo,
    )


def to_photomancy(p, band_index):
    """Photomancy [n, 8] rows: periastron phase and log scale in AU squared."""
    total_msun = p.star_mass_msun + p.planet_mass_mearth * Mearth2kg / Msun2kg
    return jnp.stack(
        (
            p.ln_period_d,
            p.ecc,
            p.cos_i,
            p.node_rad,
            p.argp_rad,
            jnp.mod(p.tau_d / jnp.exp(p.ln_period_d), 1.0),
            jnp.log(total_msun),
            jnp.log(scale_au2(p, band_index)),
        ),
        axis=-1,
    )


def truth_parameters(truth, star_mass_msun, epoch_jd):
    """Physical parameters [n_stars, S] of one population world, including fillers."""
    return PlanetParameters(
        ln_period_d=jnp.log(truth.period_d),
        ecc=truth.e,
        cos_i=truth.cos_i,
        node_rad=truth.W_rad,
        argp_rad=jnp.arctan2(truth.sin_w, truth.cos_w),
        tau_d=truth.tp_jd - epoch_jd,
        star_mass_msun=jnp.broadcast_to(star_mass_msun[:, None], truth.period_d.shape),
        planet_mass_mearth=truth.Mp_kg / Mearth2kg,
        radius_rearth=truth.Rp_AU / Rearth2AU,
        albedo=truth.albedo,
    )


class TruthCatalogRow(eqx.Module):
    """Presence [S] and named parameters [S], with albedo [S, B]."""

    present: jax.Array
    params: PlanetParameters


def catalog_template(n_slots, n_bands):
    """Shape and dtype declaration for a catalog row."""
    scalar = jax.ShapeDtypeStruct((n_slots,), F64)
    return TruthCatalogRow(
        jax.ShapeDtypeStruct((n_slots,), jnp.bool_),
        PlanetParameters(
            *([scalar] * 9), jax.ShapeDtypeStruct((n_slots, n_bands), F64)
        ),
    )


CATALOG_KIND_NAME = "truth_catalog"
CATALOG_GROUPS = ("existence", "ephemeris", "class")


def catalog_group_of(path):
    """Audience group of a leaf path or one of its indexed payload columns."""
    for field in ("present", *PlanetParameters.__dataclass_fields__):
        leaf = field if field == "present" else "params." + field
        if path == leaf or (
            path.startswith(leaf + "_")
            and all(i.isdigit() for i in path[len(leaf) + 1 :].split("_"))
        ):
            if field == "present":
                return "existence"
            return "class" if field in ("radius_rearth", "albedo") else "ephemeris"
    raise ValueError(f"unknown catalog path {path!r}")


def catalog_kind(n_slots, n_bands, open_groups=(), audience=EVERYONE):
    """A packed catalog kind; groups left closed are readable only by the scorer."""
    unknown = set(open_groups) - set(CATALOG_GROUPS)
    if unknown:
        raise ValueError(f"unknown catalog groups {sorted(unknown)}")
    return RecordKind.from_template(
        CATALOG_KIND_NAME,
        catalog_template(n_slots, n_bands),
        lambda p: audience if catalog_group_of(p) in open_groups else Audience.SCORER,
    )
