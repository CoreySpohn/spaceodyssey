"""Exception types shared across the package."""


class UnsupportedOperator(RuntimeError):
    """An implementation was asked to evaluate an operator it does not support.

    Hitting this exception means a caller used an operator that the
    implementation does not declare.
    """


class OutOfRangeError(ValueError):
    """A certificate tier was evaluated outside its validated range.

    Tiers never raise inside ``jit``; they return an in-range flag, and the
    host-side check that reads it raises this.
    """


class UnsupportedProfile(ValueError):
    """A profile requires quantities, priors, or flux units a model cannot supply."""
