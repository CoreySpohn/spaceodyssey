"""Reference claims and profile variants for exoEarth candidates.

The conservative zone (0.95 to 1.67 Earth-equivalent insolation distance)
is the exoEarth-candidate zone of Stark et al. (2019), numerically the
conservative band of Kasting et al. (1993). The radius bounds are those of
Stark et al. (2019): the lower edge 0.8 a^(-1/2) Earth radii with a in
Earth-equivalent insolation distance, equivalently 0.8 S^(1/4) with S the
insolation relative to Earth, and the upper edge 1.4 Earth radii (Kopparapu
et al. 2018). The lower edge takes only its slope from Zahnle and Catling
(2017), whose cosmic shoreline is an empirical boundary, insolation
proportional to escape velocity to the fourth power, drawn by eye in present-day
bolometric insolation relative to Earth, with no stated coefficient and no
radius form. The coefficient 0.8 and the radius form come from Kopparapu et
al. (2018) and Stark et al. (2019), who assume a common bulk density, so that
escape velocity is proportional to radius. A rocky mass-radius relation
(mass proportional to radius^3.7) would give S^0.185 instead, and a shoreline
in cumulative XUV insolation would raise the edge around low-luminosity hosts;
the community definition is kept unchanged. The
optimistic zone (0.75 to 1.77) is the recent-Venus and early-Mars limits of
Kopparapu et al. (2013). Zone edges scale with stellar luminosity only; the
host-temperature dependence of the edges (Kopparapu et al. 2013) is not
modeled. Distances are in Earth-equivalent insolation distance and radii in
Earth radii.
The water claim uses a log10 volume mixing ratio threshold of -4.
Compatible truth models must declare a prior set with id
``exoearth-reference@1``. Profile names are free-text display labels.
"""

import math
from dataclasses import replace

from spaceodyssey.profile import Claim, Edge, Profile

# Reference definition: skyscapes.habitable_zone, HZ_INNER_EEID and
# HZ_OUTER_EEID here, OPTIMISTIC_HZ_INNER_EEID and OPTIMISTIC_HZ_OUTER_EEID in
# OPTIMISTIC_ZONE. The values stay literal so profiles do not import skyscapes;
# tests/test_habitable_zone_parity.py keeps the two equal.
HABITABLE_ZONE = Claim(
    "habitable_zone",
    (
        Edge("ln_a_eeid", math.log(0.95), "above"),
        Edge("ln_a_eeid", math.log(1.67), "below"),
    ),
    counted="positive",
)
RADIUS_CLASS = Claim(
    "radius_class",
    (
        Edge("ln_rp_sqrt_a_eeid", math.log(0.8), "above"),
        Edge("ln_rp_re", math.log(1.4), "below"),
    ),
    counted="positive",
)
WATER = Claim("water", (Edge("log10_h2o_vmr", -4.0, "above"),), counted="either")

EXOEARTH_CANDIDATE = Profile(
    name="exoearth_candidate",
    claims=(HABITABLE_ZONE, RADIUS_CLASS, WATER),
    alpha=0.05,
    support_rule="joint",
    prior_set_id="exoearth-reference@1",
)
HABITABLE_ZONE_ONLY = replace(
    EXOEARTH_CANDIDATE, name="habitable_zone_only", claims=(HABITABLE_ZONE,)
)
WATER_ONLY = replace(EXOEARTH_CANDIDATE, name="water_only", claims=(WATER,))
ZONE_WATER = replace(
    EXOEARTH_CANDIDATE, name="zone_water", claims=(HABITABLE_ZONE, WATER)
)
FULL_POSITIVE = replace(
    EXOEARTH_CANDIDATE,
    name="full_positive",
    claims=(HABITABLE_ZONE, RADIUS_CLASS, replace(WATER, counted="positive")),
)
OPTIMISTIC_ZONE = replace(
    EXOEARTH_CANDIDATE,
    name="optimistic_zone",
    claims=(
        replace(
            HABITABLE_ZONE,
            edges=(
                Edge("ln_a_eeid", math.log(0.75), "above"),
                Edge("ln_a_eeid", math.log(1.77), "below"),
            ),
        ),
        RADIUS_CLASS,
        WATER,
    ),
)
ALPHA_BAND = (
    replace(EXOEARTH_CANDIDATE, name="exoearth_candidate_alpha_001", alpha=0.01),
    EXOEARTH_CANDIDATE,
    replace(EXOEARTH_CANDIDATE, name="exoearth_candidate_alpha_010", alpha=0.10),
)

__all__ = [
    "ALPHA_BAND",
    "EXOEARTH_CANDIDATE",
    "FULL_POSITIVE",
    "HABITABLE_ZONE",
    "HABITABLE_ZONE_ONLY",
    "OPTIMISTIC_ZONE",
    "RADIUS_CLASS",
    "WATER",
    "WATER_ONLY",
    "ZONE_WATER",
]
