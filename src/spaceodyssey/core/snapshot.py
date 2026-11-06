"""Snapshot."""

import dataclasses
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from spaceodyssey.core.enums import I64, Audience, B, RecordStatus, version_slot
from spaceodyssey.core.records import LOG_INT_FIELDS, RecordLog, payload_columns


class MeanBelief(eqx.Module):
    """Per-subject means, sample counts, and auxiliary analysis state."""

    mean: jax.Array
    n: jax.Array
    aux: jax.Array


def named_leaves(tree):
    """Map each dotted field path of a pytree to its leaf."""
    paths, _ = jax.tree_util.tree_flatten_with_path(tree)
    return {
        jax.tree_util.keystr(p, simple=True, separator="."): leaf for p, leaf in paths
    }


def project_mean(belief):
    """The trace projection of a `MeanBelief`: its means and auxiliary state."""
    return belief.mean, belief.aux


def _as_shapes(tree):
    """The shapes of a pytree of arrays or of `jax.ShapeDtypeStruct` values."""
    return jax.tree_util.tree_map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), tree
    )


class BeliefLayout(eqx.Module):
    """How the core reads an observer's two belief types.

    `belief` and `summary` are the shapes (pytrees of `jax.ShapeDtypeStruct`, taken
    from example values when given those) of the full belief, held in `Carry.belief`,
    and of the policy summary, held in `Carry.pbelief` and the only belief a policy
    reads. `count_leaf` names the integer leaf, present with one shape in both, that
    counts the acquisitions folded per subject; the snapshot fingerprint reads it.
    `project` maps a full belief to the pair the trace records as `belief_mean`
    (F64[n_subjects]) and `belief_aux` (F64[n_an]); `project_summary` does the same
    for a summary and defaults to `project`. A leaf of zero size is left out of the
    comparator names, so a leaf whose capacity is zero changes no canonical history.

    The projections are static and compared by identity. The core builds a layout
    inside traced code (`observer.layout(caps)`); a layout passed to a compiled
    function recompiles it unless its projections are module-level functions.
    """

    belief: eqx.Module = eqx.field(static=True, converter=_as_shapes)
    summary: eqx.Module = eqx.field(static=True, converter=_as_shapes)
    count_leaf: str = eqx.field(static=True)
    project: Callable[[eqx.Module], tuple[jax.Array, jax.Array]] = eqx.field(
        static=True
    )
    project_summary: Callable[[eqx.Module], tuple[jax.Array, jax.Array]] | None = (
        eqx.field(default=None, static=True)
    )

    def __check_init__(self):
        """Refuse a layout whose count leaf is missing, not integer, or mismatched."""
        if not isinstance(self.count_leaf, str) or not self.count_leaf:
            raise ValueError("a belief layout needs a count leaf")
        belief, summary = named_leaves(self.belief), named_leaves(self.summary)
        for which, leaves in (("belief", belief), ("summary", summary)):
            leaf = leaves.get(self.count_leaf)
            if leaf is None:
                raise ValueError(
                    f"the count leaf {self.count_leaf!r} is not a leaf of the {which}"
                )
            if not jnp.issubdtype(leaf.dtype, jnp.integer):
                raise ValueError(f"the count leaf of the {which} must be an integer")
        if belief[self.count_leaf].shape != summary[self.count_leaf].shape:
            raise ValueError("the count leaf has different shapes in the two types")

    @property
    def belief_leaves(self):
        """Comparator names of the full belief: its leaves of nonzero size."""
        return _sized(self.belief)

    @property
    def summary_leaves(self):
        """Comparator names of the policy summary: its leaves of nonzero size."""
        return _sized(self.summary)

    def count(self, belief):
        """The count leaf of a full belief or a summary."""
        return named_leaves(belief)[self.count_leaf]

    def summary_projection(self, summary):
        """`(mean, aux)` of a policy summary, by `project_summary` or `project`."""
        if self.project_summary is None:
            return self.project(summary)
        return self.project_summary(summary)


def _sized(shapes):
    leaves = named_leaves(shapes)
    return tuple(name for name, leaf in leaves.items() if np.prod(leaf.shape) > 0)


def mean_layout(belief, summary):
    """The layout of beliefs that carry `MeanBelief`'s `mean`, `n` and `aux`.

    `belief` and `summary` are shape pytrees (or example values) of the two types;
    the count leaf is `n` and both projections read `mean` and `aux`.
    """
    return BeliefLayout(
        belief=belief, summary=summary, count_leaf="n", project=project_mean
    )


class Snapshot(eqx.Module):
    """What a policy sees: released rows compacted to the front, zeros elsewhere.

    `belief` is the audience's belief: the policy summary for the policy, the full
    belief for the observer.
    """

    log: RecordLog
    n_rows: jax.Array
    belief: eqx.Module
    tick: jax.Array
    n_dec: jax.Array
    release_version: jax.Array  # releases this audience has received so far


class EffectTable(eqx.Module):
    """A decision's fixed-capacity effect rows.

    For Cancel, tick is a generation offset: 0 names the live generation and -1 a stale
    one. For Wait it is the wake tick. For Stop, scope, disposition, and lane identify
    what stops.
    """

    kind: jax.Array
    subject: jax.Array
    channel_index: jax.Array
    group: jax.Array
    tick: jax.Array
    lane: jax.Array
    scope: jax.Array
    disposition: jax.Array
    disruptive: jax.Array  # reserved: asks to break a frozen horizon


def effects_none(caps):
    """Return an effect table with no active rows."""
    z = jnp.zeros(caps.n_effects, I64)
    return EffectTable(
        **{n: z for n in EFFECT_FIELDS if n != "disruptive"},
        disruptive=jnp.zeros(caps.n_effects, B),
    )


def visible_rows(log, audience):
    """Select released policy rows or accepted observer rows by audience.

    A row held back (`RecordStatus.UNRELEASED`) is never visible, and neither is a row
    whose own audience leaves the reader out (a policy-only answer is absent from the
    observer's view).
    """
    return (
        jnp.where(
            audience == Audience.POLICY, log.released_tick >= 0, log.accepted_tick >= 0
        )
        & log.valid
        & (log.status != RecordStatus.UNRELEASED)
        & ((log.audience & int(audience)) != 0)
    )


def _superseded_in_view(log, vis):
    """Revision that replaced each row, counting only rows the audience can see."""
    same = (log.acquisition_id[:, None] == log.acquisition_id[None, :]) & vis[None, :]
    newer = same & (log.revision[None, :] > log.revision[:, None])
    return jnp.max(jnp.where(newer, log.revision[None, :], -1), axis=1)


def snapshot_view(carry, audience, caps):
    """Compact visible rows and zero every invisible column.

    The stable sort preserves public row positions when private rows are inserted.
    Settlement identities are hidden because event ids reveal private queue activity.
    A payload column the audience may not read (a scorer-only column, or any column
    the kind does not declare for it) is zero in every row, and `superseded_by` counts
    only revisions this audience can see.
    """
    log = carry.log
    vis = visible_rows(log, audience)
    order = jnp.argsort(~vis, stable=True)
    cols = {}
    for name in (*LOG_INT_FIELDS, "cost", "charged", "payload", "valid"):
        col = getattr(log, name)
        if name == "superseded_by":
            col = _superseded_in_view(log, vis)
        if name == "payload":
            col = jnp.where(payload_columns(caps, log.kind, audience), col, 0.0)
        mask = vis if col.ndim == 1 else vis[:, None]
        cols[name] = jnp.where(mask, col, jnp.zeros_like(col))[order]
    # Settlement ids shift when private events are queued; hide them from policies.
    cols["acquisition_id"] = jnp.zeros_like(cols["acquisition_id"])
    return Snapshot(
        log=RecordLog(**cols),
        n_rows=vis.sum().astype(I64),
        belief=carry.pbelief if audience == Audience.POLICY else carry.belief,
        tick=carry.tick,
        n_dec=carry.n_dec,
        release_version=carry.release_version[version_slot(audience)],
    )


_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MUL1 = np.uint64(0xBF58476D1CE4E5B9)
_MUL2 = np.uint64(0x94D049BB133111EB)


def _mix(x):
    """The splitmix64 finalizer on uint64 arrays (wrapping arithmetic)."""
    x = x + jnp.asarray(_GOLDEN, jnp.uint64)
    x = (x ^ (x >> 30)) * jnp.asarray(_MUL1, jnp.uint64)
    x = (x ^ (x >> 27)) * jnp.asarray(_MUL2, jnp.uint64)
    return x ^ (x >> 31)


def _digest_step(h, value):
    """Reuse one scan body across eager fingerprint calls."""
    return _mix(h ^ value), None


def _digest(a, salt):
    """Order-sensitive 64-bit digest of an integer or boolean array."""
    flat = jnp.ravel(a).astype(I64).astype(jnp.uint64)
    seed = _mix(jnp.asarray(np.uint64(salt), jnp.uint64))
    h, _ = jax.lax.scan(_digest_step, seed, flat)
    return _mix(h ^ jnp.asarray(flat.size, jnp.uint64))


def digest_ints(words, salt=0):
    """Order-sensitive 64-bit digest of an integer array, as an int64."""
    h = _digest(jnp.asarray(words), salt)
    return jax.lax.bitcast_convert_type(h, I64)


def snapshot_fingerprint(view, layout):
    """Identity of the decision snapshot a view holds, as an int64.

    The digest covers the integer content of the view only: every integer column of
    the visible rows, the row count, the belief's count leaf (named by the observer's
    `BeliefLayout`), the tick, the decision index and the release version. It is
    taken from the masked view, so a row, column or event the audience cannot see
    leaves it unchanged. Floating-point content is left out on purpose: it is computed
    from the prefix the integers name, and two executors may differ in it by rounding,
    which must not make their identities differ.
    """
    log = view.log
    count = layout.count(view.belief)
    leaves = [getattr(log, n) for n in LOG_INT_FIELDS] + [log.valid]
    leaves += [view.n_rows, count, view.tick, view.n_dec, view.release_version]
    # Tag field boundaries, then chain all words in one scan.
    words = []
    for i, leaf in enumerate(leaves):
        flat = jnp.ravel(jnp.asarray(leaf)).astype(I64)
        words.extend((jnp.asarray([i + 1, flat.size], I64), flat))
    h = _digest(jnp.concatenate(words), len(leaves))
    return jax.lax.bitcast_convert_type(h, I64)


EFFECT_FIELDS = tuple(f.name for f in dataclasses.fields(EffectTable))
