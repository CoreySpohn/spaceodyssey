"""One bounded executor configuration for ladder and truth-boundary checks."""

import jax
import jax.numpy as jnp
import numpy as np

from spaceodyssey.core.enums import RunStatus
from spaceodyssey.studies import bracket as br


def run(study):
    """Execute fresh seed-zero worlds with a small, declared refit shape."""
    boundary = br.bracket_boundary(
        study,
        lanes=1,
        n_grid=study.observer.prior.available.shape[1],
        n_first=8,
        n_estimate_systems=8,
        n_particles=8,
        n_round=8,
        n_rounds=2,
        n_centers=2,
        n_constants=2,
        max_rounds=2,
        forecast_quantiles=(0.5,),
    )
    source = study.world_source
    roots = jnp.stack(
        [jax.random.fold_in(jax.random.PRNGKey(0), i) for i in range(source.n_worlds)]
    )
    result = br.BoundExecutor().run_worlds(
        study, source.take(jnp.arange(source.n_worlds)), roots, 3000, refit=boundary
    )
    jax.block_until_ready(result)
    np.testing.assert_array_equal(result.status, RunStatus.COMPLETE)
    return result
