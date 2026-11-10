"""Shared build of the bracket tests: two stored stars, two worlds, the oracle arm.

The fixture is one ExoVista universe of two stars (`tests/fixtures/exovista`); the
second world is the first with every period scaled, so the two worlds differ in
what the records show.
"""

import pathlib

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from spaceodyssey.adapters.jaxedith import ObservingModeChannel
from spaceodyssey.adapters.photomancy import GRID_SPACING_D, N_GRID, DecisionSummary
from spaceodyssey.adapters.skyscapes import ExoVistaWorldSource
from spaceodyssey.studies import bracket as br

FIXTURE = pathlib.Path(__file__).parents[1] / "fixtures" / "exovista"
N_SLOTS, N_WORLDS = 6, 2
HORIZON_D, BUDGET_D = 120.0, 2.0
N_STEPS = 3000
PERIOD_SCALE = 1.07


def fixture_files():
    """The two stars' files of the one fixture universe, nearest star first."""
    return sorted(FIXTURE.glob("*-HIP_*.fits"))


def fixture_source(n_worlds=N_WORLDS):
    """The fixture as a world source; later worlds scale every period."""
    files = fixture_files()
    source = ExoVistaWorldSource.from_files([files] * n_worlds, n_slots=N_SLOTS)
    scale = jnp.asarray([PERIOD_SCALE**w for w in range(n_worlds)])[:, None, None]
    return eqx.tree_at(
        lambda s: s.population.period_d,
        source,
        source.population.period_d * scale,
    )


def fixture_channel(source, **kw):
    """The observing-mode channel of the fixture stars."""
    return ObservingModeChannel.from_stars(
        source.stars, n_slots=N_SLOTS, epoch_jd=source.epoch_jd, **kw
    )


def prior_summary(channel, n_stars, p_new=0.0):
    """The first-tier summary the observer starts from (`p_new` everywhere)."""
    summary = DecisionSummary.zeros(
        n_stars, N_SLOTS, N_GRID, channel.n_modes, channel.n_block_modes, n_an=1
    )
    return eqx.tree_at(lambda x: x.p_new, summary, summary.p_new + p_new)


def build(
    run=True,
    horizon_d=HORIZON_D,
    budget_d=BUDGET_D,
    arm="oracle",
    p_new=0.0,
    **settings,
):
    """`(source, study, executor, truths, roots, run)` of `arm` (the oracle by default).

    `p_new` is the prior chance of a new planet per star, node and mode. `settings` go
    to the arm's boundary (a posterior arm needs `occupancy`).
    """
    source = fixture_source()
    channel = fixture_channel(source)
    n_stars = len(source.stars)
    needed = np.asarray([float(channel.block_time(s)) for s in range(n_stars)])
    block_mode = [
        next((i for i, d in enumerate(channel.block_days) if d >= t), 2) for t in needed
    ]
    study = br.bracket_study(
        source,
        channel,
        arm=arm,
        horizon_d=horizon_d,
        budget_d=budget_d,
        observer=br.SummaryObserver(prior=prior_summary(channel, n_stars, p_new)),
        policy=br.BracketPolicy(block_mode, budget_d, N_SLOTS),
    )
    executor = br.BoundExecutor(boundary=br.bracket_boundary(study, arm, **settings))
    roots = jnp.stack(
        [jax.random.fold_in(jax.random.PRNGKey(1), i) for i in range(N_WORLDS)]
    )
    truths = source.take(jnp.arange(N_WORLDS))
    out = executor.run_worlds(study, truths, roots, N_STEPS) if run else None
    return source, study, executor, truths, roots, out


__all__ = ["GRID_SPACING_D", "build", "fixture_source"]
