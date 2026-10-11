"""Tests for meaning-keyed random streams."""

import jax.numpy as jnp
import jax.random as jr

from spaceodyssey.rng import Purpose, stream


def draws(coordinates, root=0):
    """Map (target, epoch, purpose) coordinates to their stream keys."""
    return {coords: stream(root, *coords).tolist() for coords in coordinates}


class TestStream:
    """Keys depend on coordinates, never on request order."""

    def test_deterministic(self):
        """Equal coordinates give equal keys."""
        a = stream(0, 7, 3, Purpose.MEASUREMENT)
        b = stream(0, 7, 3, Purpose.MEASUREMENT)
        assert jnp.array_equal(a, b)

    def test_every_coordinate_matters(self):
        """Changing any coordinate changes the key."""
        base = stream(0, 7, 3, Purpose.MEASUREMENT)
        assert not jnp.array_equal(base, stream(1, 7, 3, Purpose.MEASUREMENT))
        assert not jnp.array_equal(base, stream(0, 8, 3, Purpose.MEASUREMENT))
        assert not jnp.array_equal(base, stream(0, 7, 4, Purpose.MEASUREMENT))
        assert not jnp.array_equal(base, stream(0, 7, 3, Purpose.DETECTOR))

    def test_order_invariance(self):
        """A permuted request sequence yields identical per-coordinate keys.

        This is the property the paired-seed comparison design rests on:
        after two adaptive runs diverge, each still draws the same randomness
        for the same (target, epoch, purpose).
        """
        coordinates = [
            (t, e, p)
            for t in (1, 2, 3)
            for e in (0, 1)
            for p in (Purpose.MEASUREMENT, Purpose.DETECTOR)
        ]
        forward = draws(coordinates)
        backward = draws(list(reversed(coordinates)))
        assert forward == backward

    def test_int_and_key_roots_agree(self):
        """An integer root and its PRNGKey give the same streams."""
        from_int = stream(0, 5, 2, Purpose.POLICY)
        from_key = stream(jr.PRNGKey(0), 5, 2, Purpose.POLICY)
        assert jnp.array_equal(from_int, from_key)
