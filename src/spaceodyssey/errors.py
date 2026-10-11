"""Exception types shared across the package."""


class UnsupportedOperator(RuntimeError):
    """An implementation was asked to evaluate an operator it does not support.

    Declared operator support lives in registry metadata and is checked at
    configuration-validation time; hitting this exception at run time means a
    configuration bypassed validation.
    """


class WiringError(RuntimeError):
    """A configuration wires together incompatible seam implementations."""


class RegistryError(RuntimeError):
    """A seam registry lookup or registration failed."""
