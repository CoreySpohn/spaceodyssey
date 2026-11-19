"""The sealed world: the truth that rows are drawn from, kept out of every belief.

A world holds one point state per star. Rows come out of it only through ``observe``
(an action under a model) and ``reveal`` (a grant); ``unseal`` hands the truth to the
scorer. The noise of a row is keyed by the row's identity, never by call order.
"""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, Int, PRNGKeyArray

from spaceodyssey.belief import Belief
from spaceodyssey.types import AbstractObservationModel, Action, Atoms, Grant, Pool, Row


def row_key(
    key: PRNGKeyArray, star: Int[ArrayLike, ""], index: Int[ArrayLike, ""]
) -> PRNGKeyArray:
    """The key of one row, folded from the run key, the star and the row's index.

    The noise of a row depends on these three only, so the same row built in any order
    draws the same noise.

    Args:
        key: the run key.
        star: star index of the row.
        index: index of the row in the log.

    Returns:
        The row's key.
    """
    return jax.random.fold_in(jax.random.fold_in(key, star), index)


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
    ) -> Row:
        """The row an action returns: the model's payload drawn from the true star.

        The host passes the model and the row's key, from ``row_key``. The row is
        written in its own model's frame: ``model`` is 0 and ``payload`` has one entry.
        Its times are placeholders: ``start_d`` and ``release_d`` are 0 and
        ``charged_d`` is the exposure. The host's ``step`` sets the times and the model
        slot (moving the payload into that slot) when it writes the row to the log. A
        stack of worlds is observed under ``eqx.filter_vmap``.

        Args:
            model: the model of the action's mode.
            action: the action.
            key: the row's key.

        Returns:
            The row.
        """
        truth = jax.tree.map(lambda leaf: leaf[action.star], self._truth)
        payload = jax.tree.map(lambda x: x[0], model.sample(truth, action, key))
        zero_d = jnp.zeros_like(action.exposure_d)
        zero_i = jnp.zeros_like(action.star)
        return Row(
            star=action.star,
            mode=action.mode,
            start_d=zero_d,
            exposure_d=action.exposure_d,
            charged_d=action.exposure_d,
            release_d=zero_d,
            refusal=zero_i,
            model=zero_i,
            payload=(payload,),
        )

    def reveal(
        self,
        grant: Grant,
        models: tuple[AbstractObservationModel, ...],
        key: PRNGKeyArray,
    ) -> Row:
        """The rows of a grant, drawn from the truth, released at time zero.

        Granted row ``i`` is drawn exactly as ``observe`` draws it, with the model in
        its slot and the key ``row_key(key, star, i)``: granted row ``i`` is row ``i``
        of the log, so the host numbers its observed rows after the grant. Each row is
        released at time zero and charged nothing, keeps the exposure of its action,
        and carries its model slot, with its payload in that slot and the other slots
        zero. The rows of each model are drawn together under ``jax.vmap``.

        Args:
            grant: the grant.
            models: the model table of the run, static; ``grant.model`` indexes it.
            key: the run key.

        Returns:
            The granted rows, stacked along a leading axis.

        Raises:
            ValueError: if a model slot of the grant is not in ``models``.
        """
        if any(not 0 <= s < len(models) for s in grant.model):
            raise ValueError("a grant's model slot is not in the model table")
        actions = grant.actions
        n_row = len(grant.model)
        keys = jax.vmap(row_key, in_axes=(None, 0, 0))(
            key, actions.star, jnp.arange(n_row)
        )
        # One unbatched action, to read each model's payload shape (also for no rows).
        probe = jax.tree.map(lambda x: jnp.zeros(x.shape[1:], x.dtype), actions)

        def draw(model, action, k):
            return self.observe(model, action, k).payload[0]

        payload = []
        for slot, model in enumerate(models):
            shapes = jax.eval_shape(draw, model, probe, key)
            buffer = jax.tree.map(
                lambda x: jnp.zeros((n_row, *x.shape), x.dtype), shapes
            )
            rows = jnp.asarray([i for i, s in enumerate(grant.model) if s == slot], int)
            if rows.size:
                these = jax.tree.map(lambda leaf, rows=rows: leaf[rows], actions)
                drawn = jax.vmap(draw, in_axes=(None, 0, 0))(model, these, keys[rows])
                buffer = jax.tree.map(
                    lambda b, d, rows=rows: b.at[rows].set(d), buffer, drawn
                )
            payload.append(buffer)
        zero_d = jnp.zeros_like(actions.exposure_d)
        zero_i = jnp.zeros_like(actions.star)
        return Row(
            star=actions.star,
            mode=actions.mode,
            start_d=zero_d,
            exposure_d=actions.exposure_d,
            charged_d=zero_d,
            release_d=zero_d,
            refusal=zero_i,
            model=jnp.asarray(grant.model, dtype=zero_i.dtype).reshape(n_row),
            payload=tuple(payload),
        )

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
        pool = Pool(truth, flat, flat, no_mass, no_mass, flat)
        return Belief(truth, flat, flat, pool)


def draw_worlds(
    truth_prior: Callable[[PRNGKeyArray], Atoms], key: PRNGKeyArray, n_worlds: int
) -> SealedWorld:
    """Draw worlds from a truth prior, with a leading world axis.

    World ``i`` is drawn with ``fold_in(key, i)``, so it does not depend on how many
    worlds are drawn.

    Args:
        truth_prior: a function from a key to one truth (see ``SealedWorld``).
        key: the base key.
        n_worlds: number of worlds, static.

    Returns:
        A stack of ``n_worlds`` sealed worlds.
    """
    keys = jax.vmap(jax.random.fold_in, in_axes=(None, 0))(key, jnp.arange(n_worlds))
    return SealedWorld(jax.vmap(truth_prior)(keys))
