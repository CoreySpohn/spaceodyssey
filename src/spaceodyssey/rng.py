"""Random streams keyed by meaning, never by call order.

Every stochastic draw in a run derives its key from the run root and the
(target, epoch, purpose) coordinates of the draw. Two runs that share a root
therefore consume identical randomness for identical coordinates even when
their action sequences diverge, which is what keeps paired-seed comparison
arms aligned after adaptive trajectories separate.

Do not chain ``jax.random.split`` calls whose meaning depends on execution
order anywhere in this package; derive keys through :func:`stream`.
"""

import enum

import jax
import jax.random as jr


class Purpose(enum.IntEnum):
    """Why a random draw is being made.

    New purposes append to the end; renumbering existing members would
    silently change every keyed stream.
    """

    UNIVERSE = 0
    MEASUREMENT = 1
    DETECTOR = 2
    POLICY = 3
    STOPPING = 4


def stream(root, target_id: int, epoch_index: int, purpose: Purpose) -> jax.Array:
    """Derive the key for one (target, epoch, purpose) draw.

    Args:
        root: The run's root seed, as an integer or a PRNG key.
        target_id: Catalog target identifier the draw concerns.
        epoch_index: Index of the epoch within the run.
        purpose: What the randomness is for.

    Returns:
        A PRNG key that depends only on the four coordinates.
    """
    key = root if isinstance(root, jax.Array) else jr.PRNGKey(int(root))
    key = jr.fold_in(key, int(target_id))
    key = jr.fold_in(key, int(epoch_index))
    return jr.fold_in(key, int(purpose))
