"""Numbers held as arrays that remember the plain value they were built from."""

import jax
import jax.numpy as jnp
import numpy as np


class PlainArray(np.ndarray):
    """An array that carries `wire`: the Python value it was made from.

    A study holds a number as an array so that its value is traced and a new value
    reuses the compiled program. A document writes the field as the value it was built
    from (`3`, not `3.0`; an enum member, not its integer), which is what `wire` keeps;
    an array made any other way, or by arithmetic on this one, has no `wire` and is
    written as the plain numbers it holds. The wire plays no part in a compiled
    program, which sees only dtype and shape.
    """

    _wire = None  # a one-tuple holding the value, or None

    def __array_finalize__(self, obj):
        """A view or a result never inherits the wire."""
        self._wire = None

    @property
    def wire(self):
        """The value this array was built from, or `NO_WIRE`."""
        return NO_WIRE if self._wire is None else self._wire[0]


NO_WIRE = object()


def plain_array(dtype, x, *, row=False):
    """`jnp.asarray(x, dtype)` as a `PlainArray` that remembers `x`.

    With `row`, the array is flattened to one dimension. An `x` that is already a
    `PlainArray` keeps its own wire, so rebuilding a module leaves its document alone.
    """
    wire = x.wire if isinstance(x, PlainArray) else x
    out = np.array(jnp.asarray(x, dtype))
    if row:
        out = out.reshape(-1)
    out = out.view(PlainArray)
    if not (isinstance(x, (jax.Array, np.ndarray)) and wire is NO_WIRE):
        out._wire = (wire,)
    return out
