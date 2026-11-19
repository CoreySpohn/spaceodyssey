"""The sealed world: the truth that rows are drawn from, kept out of every belief.

A world holds one point state per star. Rows come out of it only through ``observe``
(an action under a model) and ``reveal`` (a grant); ``unseal`` hands the truth to the
scorer. The noise of a row is keyed by the row's identity, never by call order.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, Int, PRNGKeyArray, PyTree

from spaceodyssey.belief import Belief
from spaceodyssey.roles import AbstractRolePrior
from spaceodyssey.types import AbstractObservationModel, Action, Atoms, Grant, Pool

_ROW_KEY_TAG = 0
"""Tag folded into the run key before a row's identity."""

_WORLD_KEY_TAG = 1
"""Tag folded into the run key before a world's identity, so world ``i`` never shares a
key with the rows of star ``i``."""


def row_key(
    key: PRNGKeyArray, star: Int[ArrayLike, ""], index: Int[ArrayLike, ""]
) -> PRNGKeyArray:
    """The key of one row, folded from the run key, the star and the row's index.

    The noise of a row depends on these three only, so the same row built in any order
    draws the same noise. A row tag is folded in first, so no row shares a key with a
    world drawn from the same run key.

    Args:
        key: the run key.
        star: star index of the row.
        index: index of the row in the log.

    Returns:
        The row's key.
    """
    tagged = jax.random.fold_in(key, _ROW_KEY_TAG)
    return jax.random.fold_in(jax.random.fold_in(tagged, star), index)


class SealedWorld(eqx.Module):
    """The truth of a run, one point state per star.

    ``_truth`` is an ``Atoms`` whose leaves carry a leading star axis ``S`` and then one
    draw (``A = 1``); a stack of worlds adds a world axis in front. On construction the
    single draw's cloud weights are set to zero (weight one) and its gas block is
    checked to be a point state (``x_var = 0``).
    """

    _truth: Atoms

    def __init__(self, truth: Atoms):
        """Seal a truth.

        Args:
            truth: the true state, leaves of shape ``(..., S, 1, ...)``; the gas block
                has shape ``(..., S, 1, K, G)``.

        Raises:
            ValueError: if the gas block has no star axis or more than one draw.
            EquinoxRuntimeError: if a gas of the truth has nonzero variance.
        """
        if truth.x_mean.ndim < 4 or truth.x_mean.shape[-3] != 1:
            raise ValueError("a truth holds one draw per star, gas block (S, 1, K, G)")
        x_var = eqx.error_if(
            truth.x_var, truth.x_var != 0, "the truth is a point state (x_var = 0)"
        )
        zero = jnp.zeros_like(truth.cloud.log_weight)
        self._truth = eqx.tree_at(
            lambda a: (a.x_var, a.cloud.log_weight, a.cloud.log_ref_density),
            truth,
            (x_var, zero, zero),
        )

    def observe(
        self, model: AbstractObservationModel, action: Action, key: PRNGKeyArray
    ) -> PyTree:
        """The payload an action returns, drawn from the true star with its model.

        The host passes the model and the row's key, from ``row_key``, and writes the
        row: its times, charge and model slot are the host's bookkeeping, not the
        world's. A stack of worlds is observed under ``eqx.filter_vmap``.

        Args:
            model: the model of the action's mode.
            action: the action.
            key: the row's key.

        Returns:
            The payload, a dict of named arrays.
        """
        truth = jax.tree.map(lambda leaf: leaf[action.star], self._truth)
        return jax.tree.map(lambda x: x[0], model.sample(truth, action, key))

    def reveal(
        self,
        grant: Grant,
        models: tuple[AbstractObservationModel, ...],
        key: PRNGKeyArray,
    ) -> tuple[PyTree, ...]:
        """The payloads of a grant's rows, drawn from the truth.

        Granted row ``i`` is drawn exactly as ``observe`` draws it, with the model in
        its slot and the key ``row_key(key, star, i)``: granted row ``i`` is row ``i``
        of the log, so the host numbers its observed rows after the grant. The rows of
        each model are drawn together under ``jax.vmap``. ``grant_rows`` builds the
        log rows from the payloads.

        Args:
            grant: the grant.
            models: the model table of the run, static; ``grant.model_slot`` indexes
                it.
            key: the run key.

        Returns:
            One payload buffer per model, each a dict of arrays with a leading axis
            over the granted rows; row ``i`` holds its payload in its own model's
            buffer and zeros in the others.

        Raises:
            ValueError: if a model slot of the grant is not in ``models``.
        """
        if any(not 0 <= s < len(models) for s in grant.model_slot):
            raise ValueError("a grant's model slot is not in the model table")
        actions = grant.actions
        n_row = len(grant.model_slot)
        keys = jax.vmap(row_key, in_axes=(None, 0, 0))(
            key, actions.star, jnp.arange(n_row)
        )
        # One unbatched action, to read each model's payload shape (also for no rows).
        probe = jax.tree.map(lambda x: jnp.zeros(x.shape[1:], x.dtype), actions)

        payload = []
        for slot, model in enumerate(models):
            shapes = jax.eval_shape(self.observe, model, probe, key)
            buffer = jax.tree.map(
                lambda x: jnp.zeros((n_row, *x.shape), x.dtype), shapes
            )
            rows = jnp.asarray(
                [i for i, s in enumerate(grant.model_slot) if s == slot], int
            )
            if rows.size:
                these = jax.tree.map(lambda leaf, rows=rows: leaf[rows], actions)
                drawn = jax.vmap(self.observe, in_axes=(None, 0, 0))(
                    model, these, keys[rows]
                )
                buffer = jax.tree.map(
                    lambda b, d, rows=rows: b.at[rows].set(d), buffer, drawn
                )
            payload.append(buffer)
        return tuple(payload)

    def unseal(self) -> Belief:
        """The truth as a belief with one role: one draw of weight one, an empty pool.

        Only the scorer calls this. The belief keeps the leading star axis, so a claim
        is read per star with ``jax.vmap``. The pool's single mark is the truth itself
        with zero expected mass, so it never counts.

        Returns:
            A belief whose leaves have a leading star axis.
        """
        truth = self._truth
        lead = truth.cloud.log_weight.shape[:-1]
        flat = jnp.zeros((*lead, 1, 1))
        no_mass = jnp.full((*lead, 1), -jnp.inf)
        pool = Pool(truth, flat, flat, no_mass)
        return Belief(truth, flat, flat, pool)


def draw_worlds(
    prior: AbstractRolePrior, stars, key: PRNGKeyArray, n_worlds: int
) -> SealedWorld:
    """Draw worlds from the truth's role prior, with a leading world axis.

    Star ``s`` of world ``i`` is ``prior.sample`` with the key folded from the run key,
    a world tag, ``i`` and ``s``, so world ``i`` does not depend on how many worlds are
    drawn and shares no key with any row.

    Args:
        prior: the truth's role prior.
        stars: the stars, a pytree whose leaves have a leading star axis ``S``.
        key: the run key.
        n_worlds: number of worlds, static.

    Returns:
        A stack of ``n_worlds`` sealed worlds over ``S`` stars.
    """
    n_star = jax.tree.leaves(stars)[0].shape[0]
    tagged = jax.random.fold_in(key, _WORLD_KEY_TAG)

    def world(i):
        keys = jax.vmap(jax.random.fold_in, in_axes=(None, 0))(
            jax.random.fold_in(tagged, i), jnp.arange(n_star)
        )
        return jax.vmap(prior.sample)(keys, stars)

    return SealedWorld(jax.vmap(world)(jnp.arange(n_worlds)))
