"""The sealed world: the truth that rows are drawn from, kept out of every belief.

A world holds one point state per star. Rows come out of it only through ``observe``
(an action under a model) and ``reveal`` (a grant); ``unseal`` hands the truth to the
scorer. The noise of a row is keyed by the row's identity, never by call order.

One run key serves a run: ``run_keys`` splits it into a world key, from which
``draw_worlds`` draws the truth, a noise key, from which every row's noise is folded,
and a policy key, from which every decision's key is folded. The three are tagged
apart, so no truth shares a key with any row or decision, and no row with a decision.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, Int, PRNGKeyArray, PyTree

from spaceodyssey.design import Mode
from spaceodyssey.roles import AbstractRolePrior
from spaceodyssey.types import (
    AbstractObservationModel,
    Action,
    Atoms,
    Grant,
    WeightedCloud,
)

_NOISE_KEY_TAG = 0
"""Tag folded into the run key to give the noise key."""

_WORLD_KEY_TAG = 1
"""Tag folded into the run key to give the world key, so world ``i`` never shares a
key with the rows of star ``i``."""

_POLICY_KEY_TAG = 2
"""Tag folded into the run key to give the policy key, so a decision never shares a
key with a world or a row."""


def run_keys(key: PRNGKeyArray) -> tuple[PRNGKeyArray, PRNGKeyArray, PRNGKeyArray]:
    """Split a run key into its world key, its noise key and its policy key.

    ``draw_worlds`` takes the run key and draws from the world key; ``execute`` takes
    the same run key, draws every row's noise from the noise key and hands each
    decision a key folded from the policy key. None depends on another or on which
    runs first.

    Args:
        key: the run key.

    Returns:
        The world key, the noise key and the policy key.
    """
    return (
        jax.random.fold_in(key, _WORLD_KEY_TAG),
        jax.random.fold_in(key, _NOISE_KEY_TAG),
        jax.random.fold_in(key, _POLICY_KEY_TAG),
    )


def row_key(
    key: PRNGKeyArray, star: Int[ArrayLike, ""], index: Int[ArrayLike, ""]
) -> PRNGKeyArray:
    """The key of one row, folded from the noise key, the star and the row's index.

    The noise of a row depends on these three only, so the same row built in any order
    draws the same noise.

    Args:
        key: the run's noise key (``run_keys``).
        star: star index of the row.
        index: index of the row in the log.

    Returns:
        The row's key.
    """
    return jax.random.fold_in(jax.random.fold_in(key, star), index)


class SealedWorld(eqx.Module):
    """The truth of a run, one point state per star.

    ``_truth`` is an ``Atoms`` whose leaves carry a leading star axis ``S`` and then one
    draw (``A = 1``) of ``K`` planets; a stack of worlds adds a world axis in front. On
    construction the single draw is given unit weight (its log weight and its reference
    density zero, on the truth's chart) and its gas block is checked to be a point
    state (``x_var = 0``).
    """

    _truth: Atoms

    def __init__(self, truth: Atoms):
        """Seal a truth.

        Args:
            truth: the true state, leaves of shape ``(..., S, 1, ...)``; the gas block
                has shape ``(..., S, 1, K, G)``.

        Raises:
            ValueError: if the gas block has no star axis or more than one draw, or
                the truth declares no chart.
            EquinoxRuntimeError: if a gas of the truth has nonzero variance.
        """
        if truth.x_mean.ndim < 4 or truth.x_mean.shape[-3] != 1:
            raise ValueError("a truth holds one draw per star, gas block (S, 1, K, G)")
        x_var = eqx.error_if(
            truth.x_var, truth.x_var != 0, "the truth is a point state (x_var = 0)"
        )
        cloud = truth.cloud
        zero = jnp.zeros_like(cloud.log_weights)
        # Built through the constructor, so the cloud's own checks run on the truth.
        unit = WeightedCloud(
            cloud.samples,
            zero,
            cloud.evidence,
            zero,
            chart=cloud.chart,
            epoch=cloud.epoch,
        )
        self._truth = Atoms(unit, truth.x_mean, x_var, truth.exists)

    def observe(
        self, key: PRNGKeyArray, model: AbstractObservationModel, action: Action
    ) -> PyTree:
        """The payload an action returns, drawn from the true star with its model.

        The host passes the model and the row's key, from ``row_key``, and writes the
        row: its times, charge and model slot are the host's bookkeeping, not the
        world's. A stack of worlds is observed under ``eqx.filter_vmap``.

        Args:
            key: the row's key.
            model: the model of the action's mode.
            action: the action.

        Returns:
            The payload, a dict of named arrays.
        """
        truth = jax.tree.map(lambda leaf: leaf[action.star], self._truth)
        return jax.tree.map(lambda x: x[0], model.sample(key, truth, action))

    def reveal(
        self,
        key: PRNGKeyArray,
        grant: Grant,
        models: tuple[AbstractObservationModel, ...],
        modes: tuple[Mode, ...],
    ) -> tuple[PyTree, ...]:
        """The payloads of a grant's rows, drawn from the truth.

        Granted row ``i`` is drawn exactly as ``observe`` draws it, with the model of
        its mode and the key ``row_key(key, star, i)``: granted row ``i`` is row ``i``
        of the log, so the host numbers its observed rows after the grant. The rows of
        each model are drawn together under ``jax.vmap``. ``grant_rows`` builds the
        log rows from the payloads.

        Args:
            key: the run's noise key (``run_keys``).
            grant: the grant, its actions concrete.
            models: the design's model table, static.
            modes: the design's mode table, static; a granted action's mode indexes it.

        Returns:
            One payload buffer per model, each a dict of arrays with a leading axis
            over the granted rows; row ``i`` holds its payload in its own model's
            buffer and zeros in the others.

        Raises:
            ValueError: if a granted row's mode is not in ``modes`` or its model slot
                is not in ``models``.
        """
        actions = grant.actions
        granted = [int(m) for m in jnp.ravel(actions.mode)]
        if any(not 0 <= m < len(modes) for m in granted):
            raise ValueError("a grant's mode is not in the mode table")
        slots = [modes[m].model_slot for m in granted]
        if any(not 0 <= s < len(models) for s in slots):
            raise ValueError("a grant's model slot is not in the model table")
        n_row = len(slots)
        keys = jax.vmap(row_key, in_axes=(None, 0, 0))(
            key, actions.star, jnp.arange(n_row)
        )
        # One unbatched action, to read each model's payload shape (also for no rows).
        probe = jax.tree.map(lambda x: jnp.zeros(x.shape[1:], x.dtype), actions)

        payload = []
        for slot, model in enumerate(models):
            shapes = jax.eval_shape(self.observe, key, model, probe)
            buffer = jax.tree.map(
                lambda x: jnp.zeros((n_row, *x.shape), x.dtype), shapes
            )
            rows = jnp.asarray([i for i, s in enumerate(slots) if s == slot], int)
            if rows.size:
                these = jax.tree.map(lambda leaf, rows=rows: leaf[rows], actions)
                drawn = jax.vmap(self.observe, in_axes=(0, None, 0))(
                    keys[rows], model, these
                )
                buffer = jax.tree.map(
                    lambda b, d, rows=rows: b.at[rows].set(d), buffer, drawn
                )
            payload.append(buffer)
        return tuple(payload)

    def unseal(self) -> Atoms:
        """The truth: one draw of unit weight per star, a point state.

        Only the scorer calls this. The draws keep the leading star axis, and every
        planet slot of the truth, so a star with several planets is read whole.

        Returns:
            The truth's draws, leaves with a leading star axis.
        """
        return self._truth


def draw_worlds(
    key: PRNGKeyArray, prior: AbstractRolePrior, stars, n_worlds: int
) -> SealedWorld:
    """Draw worlds from the truth's role prior, with a leading world axis.

    Star ``s`` of world ``i`` is ``prior.sample`` with the key folded from the run's
    world key (``run_keys``), ``i`` and ``s``, so world ``i`` does not depend on how
    many worlds are drawn and shares no key with any row.

    Args:
        key: the run key.
        prior: the truth's role prior.
        stars: the stars, a pytree whose leaves have a leading star axis ``S``.
        n_worlds: number of worlds, static.

    Returns:
        A stack of ``n_worlds`` sealed worlds over ``S`` stars.
    """
    n_star = jax.tree.leaves(stars)[0].shape[0]
    world_key, _, _ = run_keys(key)

    def world(i):
        keys = jax.vmap(jax.random.fold_in, in_axes=(None, 0))(
            jax.random.fold_in(world_key, i), jnp.arange(n_star)
        )
        return jax.vmap(prior.sample)(keys, stars)

    return SealedWorld(jax.vmap(world)(jnp.arange(n_worlds)))
