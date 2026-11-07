"""What every battery check returns, and the small readers they share."""

import dataclasses

import jax
import numpy as np

from spaceodyssey.core.executors.base import index_world
from spaceodyssey.core.executors.compiled import CompiledExecutor


@dataclasses.dataclass(frozen=True)
class CheckResult:
    """The outcome of one check: a pass flag, a diagnostic and the numbers behind it.

    `diagnostic` is empty on a pass and names the first failing property otherwise.
    `detail` holds counts a caller can print or assert on, so a vacuous pass shows.
    """

    name: str
    passed: bool
    diagnostic: str = ""
    detail: dict = dataclasses.field(default_factory=dict)

    def __bool__(self):
        """A result is truthy exactly when the check passed."""
        return self.passed

    def __str__(self):
        """One line: the name, the verdict and the diagnostic."""
        verdict = "pass" if self.passed else f"FAIL: {self.diagnostic}"
        return f"{self.name}: {verdict}"


def make_result(name, failures, **detail):
    """A `CheckResult` that fails when `failures` (a list of strings) is nonempty."""
    return CheckResult(name, not failures, "; ".join(failures), dict(detail))


def default_executor(executor):
    """The compiled executor unless the caller names another."""
    return CompiledExecutor() if executor is None else executor


def bits_equal(a, b) -> bool:
    """Whether two arrays have one shape, one dtype and identical bytes."""
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def tree_bits_equal(a, b) -> bool:
    """Whether two pytrees have one structure and bitwise-equal leaves."""
    la, ta = jax.tree_util.tree_flatten(a)
    lb, tb = jax.tree_util.tree_flatten(b)
    return ta == tb and all(bits_equal(x, y) for x, y in zip(la, lb, strict=True))


def worlds_of(run):
    """The per-world (carry, trace) pairs of a batched run."""
    n = int(np.asarray(run.carry.tick).shape[0])
    return [(index_world(run.carry, w), index_world(run.trace, w)) for w in range(n)]


def log_index(carry):
    """Map each valid log row's identity to its slot.

    The identity is (subject, acquisition index, channel, revision, status, audience):
    one datum written once per audience (a split pre-run prior) is two rows. Two
    valid rows with one identity are refused with `ValueError`, never indexed as one.
    """
    log = carry.log
    out = {}
    for i in np.flatnonzero(np.asarray(log.valid)):
        key = (
            *(
                int(getattr(log, n)[i])
                for n in (
                    "subject",
                    "acquisition_index",
                    "channel",
                    "revision",
                    "status",
                )
            ),
            int(log.audience[i]),
        )
        if key in out:
            raise ValueError(
                f"log rows {out[key]} and {int(i)} share the identity {key}; a "
                "comparison cannot tell them apart"
            )
        out[key] = int(i)
    return out
