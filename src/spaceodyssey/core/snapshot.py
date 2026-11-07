"""Snapshot."""

import dataclasses
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from spaceodyssey.core.capacities import OPTIONAL
from spaceodyssey.core.enums import I64, Audience, B, RecordStatus, version_slot
from spaceodyssey.core.records import (
    LOG_INT_FIELDS,
    REFUND_COLUMNS,
    RecordLog,
    derived_rows,
    hide_refund_of,
    payload_columns,
)


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
    # The policy's view of the ledger (rows: assets, then programs, then the shared
    # row; one column per resource), as the float balance and the integer count of
    # `balance_quantum` units the fingerprint hashes. Absent (None, no leaf) unless
    # `snapshot_balances` is on, and always absent for the observer. `balance_rows`
    # names each row (columns: kind 0 asset, 1 program, 2 shared; index within the
    # kind), so two layouts with the same row count cannot show one table.
    balance: jax.Array | None = None
    balance_q: jax.Array | None = None
    balance_rows: jax.Array | None = None


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
    # The program a row charges (and, for a Stop, closes); absent (None, no leaf) unless
    # the capacities hold more than one program, so a study with one is unchanged.
    program: jax.Array | None = dataclasses.field(default=None, metadata=OPTIONAL)


def effects_none(caps):
    """Return an effect table with no active rows."""
    z = jnp.zeros(caps.n_effects, I64)
    return EffectTable(
        **{n: z for n in EFFECT_FIELDS if n != "disruptive"},
        disruptive=jnp.zeros(caps.n_effects, B),
        program=z if caps.n_programs > 1 else None,
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


def _superseded_in_view(log, vis, caps):
    """Revision that replaced each row, counting only rows the audience can see.

    A derived row neither replaces nor is replaced.
    """
    vis = vis & ~derived_rows(log, caps)
    same = (log.acquisition_id[:, None] == log.acquisition_id[None, :]) & vis[None, :]
    newer = same & (log.revision[None, :] > log.revision[:, None])
    out = jnp.max(jnp.where(newer, log.revision[None, :], -1), axis=1)
    return jnp.where(derived_rows(log, caps), -1, out)


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
            col = _superseded_in_view(log, vis, caps)
        if name == "payload":
            col = jnp.where(payload_columns(caps, log.kind, audience), col, 0.0)
        if name == "refund_of" and caps.refunds:
            col = hide_refund_of(col)
        mask = vis if col.ndim == 1 else vis[:, None]
        cols[name] = jnp.where(mask, col, jnp.zeros_like(col))[order]
    # a pedigree is shown only on the visible product rows that carry one
    cols["product"] = jnp.where(vis[:, None], log.product, 0)[order]
    # Settlement ids shift when private events are queued; hide them from policies.
    cols["acquisition_id"] = jnp.zeros_like(cols["acquisition_id"])
    if log.program is not None:
        # only the policy is shown a program id; the observer's column is all zeros
        shown = vis if audience == Audience.POLICY else jnp.zeros_like(vis)
        cols["program"] = jnp.where(shown, log.program, 0)[order]
    else:
        cols["program"] = None
    for name in REFUND_COLUMNS:
        col = getattr(log, name)
        if col is None:
            cols[name] = None
        else:
            mask = vis if col.ndim == 1 else vis[:, None]
            cols[name] = jnp.where(mask, col, jnp.zeros_like(col))[order]
    balance, balance_q, balance_rows = balance_view(carry, audience, caps)
    return Snapshot(
        log=RecordLog(**cols),
        n_rows=vis.sum().astype(I64),
        belief=carry.pbelief if audience == Audience.POLICY else carry.belief,
        tick=carry.tick,
        n_dec=carry.n_dec,
        release_version=carry.release_version[version_slot(audience)],
        balance=balance,
        balance_q=balance_q,
        balance_rows=balance_rows,
    )


def balance_view(carry, audience, caps):
    """The balance, its integer count of quanta and its row names, or all `None`.

    Only the policy is shown balances, and only while `caps.snapshot_balances` is on,
    so every other view keeps its pytree and its fingerprint. The integer is the
    balance over `caps.balance_quantum`, rounded to the nearest unit; a study that
    shows balances refuses a budget the integer could not hold (`Study`). The row
    names are the ledger's layout: assets, then programs, then the shared row.
    """
    if not caps.snapshot_balances or audience != Audience.POLICY:
        return None, None, None
    balance = carry.ledger.balance
    kind = np.repeat([0, 1, 2], [caps.n_assets, caps.n_programs, 1])
    index = np.concatenate(
        [np.arange(caps.n_assets), np.arange(caps.n_programs), np.zeros(1, int)]
    )
    rows = jnp.asarray(np.stack([kind, index], axis=1), I64)
    return balance, jnp.round(balance / caps.balance_quantum).astype(I64), rows


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


def conditional_view_leaves(view):
    """The leaves of a view that exist only where a feature does, in digest order.

    The pedigree the view shows (a width-0 column when the refresh protocol is off) and
    the program column (absent with one program), the refund index and award generation
    (absent without refunds), then the balance row names and the integer balance (absent
    unless balances are on; the float is not hashed). A feature that adds a leaf to the
    view lists it here.
    """
    program = () if view.log.program is None else (view.log.program,)
    # the refund columns that are integers; the amount is a float and is not hashed
    refunds = tuple(
        x for x in (view.log.refund_index, view.log.award_gen) if x is not None
    )
    balance = () if view.balance_q is None else (view.balance_rows, view.balance_q)
    return (view.log.product, *program, *refunds, *balance)


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
    # A leaf of a feature that is off is empty and is not listed at all: the digest
    # tags and counts every leaf it hashes, so even an empty one would move it.
    leaves += [x for x in conditional_view_leaves(view) if jnp.size(x) > 0]
    # Tag field boundaries, then chain all words in one scan.
    words = []
    for i, leaf in enumerate(leaves):
        flat = jnp.ravel(jnp.asarray(leaf)).astype(I64)
        words.extend((jnp.asarray([i + 1, flat.size], I64), flat))
    h = _digest(jnp.concatenate(words), len(leaves))
    return jax.lax.bitcast_convert_type(h, I64)


# the columns every effect table has; `program` is present only with several programs
EFFECT_FIELDS = tuple(
    f.name for f in dataclasses.fields(EffectTable) if f.name != "program"
)
