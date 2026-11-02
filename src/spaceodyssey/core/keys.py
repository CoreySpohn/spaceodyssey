"""Keys.

A draw is addressed by coordinates, never by step or batch position. The nine
coordinates are world, subject, asset, channel, acquisition, phase, branch, lineage and
purpose; `revision` and `status` ride along for record variants. The two newest
coordinates, `asset` and `branch`, are folded only when nonzero, so a key that does not
use them equals the key built before they existed.

A `Coupling` declares, per purpose, which coordinates the draws of that purpose share
across: a dropped coordinate is folded as zero, so changing it moves no draw.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from spaceodyssey.core.enums import BIG, I64, Purpose

MASK32 = (1 << 32) - 1
COORDS = (
    "world",
    "subject",
    "channel",
    "acquisition_index",
    "phase",
    "lineage",
    "purpose",
    "revision",
    "status",
    "asset",
    "branch",
)
OPTIONAL_COORDS = frozenset({"asset", "branch"})
# A nonzero optional coordinate folds its tag before its value, so asset=5 and
# branch=5 give different keys. The tags are arbitrary distinct constants.
_OPTIONAL_TAGS = {"asset": 0x6173736574, "branch": 0x6272616E6368}


class Coupling(eqx.Module):
    """Per purpose, the coordinates whose changes leave that purpose's draws alone.

    `shared` holds (purpose, coordinate names) pairs. The default shares nothing, so
    every coordinate separates draws.
    """

    shared: tuple = eqx.field(default=(), static=True)

    def __check_init__(self):
        """Reject unknown coordinate names and the identity coordinates."""
        for purpose, names in self.shared:
            Purpose(purpose)
            unknown = set(names) - set(COORDS) - {"tick", "ordinal"}
            if unknown or {"tick", "ordinal", "purpose"} & set(names):
                raise ValueError(f"Cannot share: {', '.join(sorted(unknown))}")

    def shared_for(self, purpose) -> frozenset:
        """The coordinate names shared across for one purpose."""
        out = set()
        for p, names in self.shared:
            if int(p) == int(purpose):
                out |= set(names)
        return frozenset(out)


DEFAULT_COUPLING = Coupling()


class Fork(eqx.Module):
    """A counterfactual branch: acquisitions from `from_index` on draw on `branch`.

    The default forks nothing (the start index is out of reach and the branch is zero),
    and a zero branch leaves every key as it was before forks existed.
    """

    from_index: jax.Array = eqx.field(
        default=BIG, converter=lambda x: jnp.asarray(x, I64)
    )
    branch: jax.Array = eqx.field(default=0, converter=lambda x: jnp.asarray(x, I64))

    def branch_of(self, acquisition_index) -> jax.Array:
        """The branch coordinate of an acquisition by its index."""
        return jnp.where(acquisition_index >= self.from_index, self.branch, 0)


def fold64(key, x) -> jax.Array:
    """Fold both words of a 64-bit integer, low word first."""
    lo = jnp.asarray(x & MASK32, dtype=jnp.uint32)
    hi = jnp.asarray((x >> 32) & MASK32, dtype=jnp.uint32)
    return jr.fold_in(jr.fold_in(key, lo), hi)


def key_for(root, **coords) -> jax.Array:
    """Fold the ordered coordinates, defaulting missing coordinates to zero.

    Every value may be traced. `asset` and `branch` leave the key unchanged at zero.
    """
    unknown = coords.keys() - set(COORDS)
    if unknown:
        raise TypeError(f"Unknown coordinates: {', '.join(sorted(unknown))}")
    key = root
    for name in COORDS:
        x = coords.get(name, 0)
        if name in OPTIONAL_COORDS:
            if isinstance(x, int) and x == 0:
                continue
            x = jnp.asarray(x, dtype=jnp.int64)
            tagged = fold64(fold64(key, _OPTIONAL_TAGS[name]), x)
            key = jnp.where(x != 0, tagged, key)
        else:
            key = fold64(key, x)
    return key


def draw_key(
    root, rec, purpose, *, asset=0, branch=0, lineage=0, coupling=DEFAULT_COUPLING
) -> jax.Array:
    """Key a draw by record identity and purpose, less the shared coordinates."""
    coords = {
        "world": 0,
        "subject": rec.subject,
        "channel": rec.channel,
        "acquisition_index": rec.acquisition_index,
        "phase": rec.phase,
        "lineage": lineage,
        "purpose": purpose,
        "revision": rec.revision,
        "status": rec.status,
        "asset": asset,
        "branch": branch,
    }
    for name in coupling.shared_for(purpose):
        coords[name] = 0
    return key_for(root, **coords)


def policy_key(
    root, phase, tick, ordinal, *, lineage=0, coupling=DEFAULT_COUPLING
) -> jax.Array:
    """Key a policy draw by meaning: phase, lineage, decision tick and ordinal in tick.

    The ordinal counts earlier decisions at the same tick, so a decision added at
    another tick moves no draw. A declared coupling may share `phase` or `lineage`.
    """
    shared = coupling.shared_for(Purpose.POLICY)
    coords = {"phase": phase, "lineage": lineage}
    key = fold64(root, Purpose.POLICY)
    for name in ("phase", "lineage"):
        key = fold64(key, 0 if name in shared else coords[name])
    return fold64(fold64(key, tick), ordinal)
