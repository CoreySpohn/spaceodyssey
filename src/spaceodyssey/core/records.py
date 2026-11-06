"""Records."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp

from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.enums import EVERYONE, F64, I64, Audience, B, Capacity


@dataclasses.dataclass(frozen=True)
class PayloadColumn:
    """One payload column and the audiences that may read it."""

    name: str
    audience: int

    def __post_init__(self):
        """Require a declared, known audience."""
        if self.audience is None:
            raise ValueError(f"column {self.name!r} declares no audience")
        flags = int(self.audience)
        if flags == 0 or flags & ~int(EVERYONE):
            raise ValueError(
                f"column {self.name!r} needs a nonempty audience built from "
                "Audience flags"
            )
        object.__setattr__(self, "audience", flags)


@dataclasses.dataclass(frozen=True)
class RecordKind:
    """A registered record layout: a name and its payload columns in order.

    A `derived` kind holds a product computed from other records (a summary or a
    refit product); an observer never names it as refit data.
    """

    name: str
    columns: tuple[PayloadColumn, ...]
    derived: bool = False

    def __post_init__(self):
        """Require typed columns with distinct names."""
        if not all(isinstance(c, PayloadColumn) for c in self.columns):
            raise ValueError(f"kind {self.name!r} needs PayloadColumn entries")
        names = [c.name for c in self.columns]
        if len(set(names)) != len(names):
            raise ValueError(f"kind {self.name!r} repeats a column name")

    @property
    def width(self) -> int:
        """Number of payload columns."""
        return len(self.columns)


class KindRegistry:
    """An in-process list of record kinds; a kind's index is its record `kind` value.

    Registering is the only way a layout reaches the carry: `lower` widens
    `Capacities.payload_width` to the widest kind and records every column's audience.
    """

    def __init__(self, kinds=()):
        """Register `kinds` in order."""
        self._kinds = []
        for kind in kinds:
            self.register(kind)

    def register(self, kind):
        """Add a kind and return its index."""
        if not isinstance(kind, RecordKind):
            raise TypeError("register needs a RecordKind")
        if any(k.name == kind.name for k in self._kinds):
            raise ValueError(f"kind {kind.name!r} is already registered")
        self._kinds.append(kind)
        return len(self._kinds) - 1

    def index(self, name):
        """Index of the kind registered under `name`."""
        for i, k in enumerate(self._kinds):
            if k.name == name:
                return i
        raise KeyError(name)

    def __len__(self):
        """Number of registered kinds."""
        return len(self._kinds)

    def __getitem__(self, i):
        """The kind at index `i`."""
        return self._kinds[i]

    @property
    def width(self) -> int:
        """Width of the widest registered kind."""
        return max((k.width for k in self._kinds), default=0)

    def lower(self, caps=None, observer=None):
        """Return `caps` with this registry's layouts and `observer`'s refit kinds.

        The width is the larger of `caps.payload_width` and the widest kind. A column
        a kind does not declare has no audience and is unreadable. The kind names and
        the indices of derived kinds are recorded, and the refit kinds and statuses
        the observer's `capabilities()` declare are lowered by `lower_refit`, which
        refuses a derived or unregistered kind name.
        """
        caps = Capacities() if caps is None else caps
        width = max(caps.payload_width, self.width)
        table = tuple(
            tuple(c.audience for c in k.columns) + (0,) * (width - k.width)
            for k in self._kinds
        )
        names = tuple(k.name for k in self._kinds)
        derived = tuple(i for i, k in enumerate(self._kinds) if k.derived)
        declared = None if observer is None else observer.capabilities()
        kinds, statuses = lower_refit(declared, names, derived)
        return dataclasses.replace(
            caps,
            payload_width=width,
            payload_audience=table,
            kind_names=names,
            derived_kinds=derived,
            refit_kinds=kinds,
            refit_statuses=statuses,
        )


def lower_refit(declared, kind_names, derived_kinds):
    """`(refit_kinds, refit_statuses)` as `Capacities` stores them.

    `declared` is an observer's `capabilities()` (or `None`). Each refit kind name
    becomes its registered index; a name no kind is registered under, or one that
    names a derived kind, is refused. An observer that does not refit lowers to
    empty tuples.
    """
    if declared is None or not declared.refit:
        return (), ()
    indices = []
    for name in declared.refit_kinds:
        if name not in kind_names:
            raise ValueError(f"refit kind {name!r} is not a registered record kind")
        index = kind_names.index(name)
        if index in derived_kinds:
            raise ValueError(
                f"refit kind {name!r} is derived; a product is never refit data"
            )
        indices.append(index)
    return tuple(indices), tuple(int(s) for s in declared.refit_statuses)


def payload_columns(caps, kind, audience):
    """Which payload columns of records of `kind` the `audience` may read.

    Returns bool[..., P] shaped like `kind` plus a column axis. A caps with no lowered
    registry leaves every column readable; a kind index the registry lacks has none.
    """
    kind = jnp.asarray(kind, I64)
    table = caps.payload_audience
    if not table:
        return jnp.ones((*kind.shape, caps.payload_width), B)
    flags = jnp.asarray(table, I64)
    known = (kind >= 0) & (kind < flags.shape[0])
    row = flags[jnp.clip(kind, 0, flags.shape[0] - 1)]
    return ((row & int(audience)) != 0) & known[..., None]


def masked_record(rec, caps, audience):
    """Hide queue-derived identities and payload columns the audience cannot read."""
    keep = payload_columns(caps, rec.kind, audience)
    return dataclasses.replace(
        rec,
        acquisition_id=jnp.zeros_like(rec.acquisition_id),
        payload=jnp.where(keep, rec.payload, 0.0),
    )


class Record(eqx.Module):
    """One record with provenance, timing, a cost vector, and a typed payload."""

    kind: jax.Array
    subject: jax.Array
    channel: jax.Array
    channel_index: jax.Array
    lane: jax.Array
    acquisition_index: jax.Array
    start_tick: jax.Array
    end_tick: jax.Array
    available_tick: jax.Array
    status: jax.Array
    phase: jax.Array
    acquisition_id: jax.Array
    revision: jax.Array  # product revision of the acquisition, 0 for the first
    released_tick: jax.Array  # release tick, -1 until released
    accepted_tick: jax.Array  # tick the observer accepted it, -1 until then
    superseded_by: jax.Array  # revision that replaced it, -1 if none
    audience: jax.Array  # Audience flags the record is released to
    source: jax.Array  # channel id
    vintage: jax.Array  # phase index at admission
    config_id: jax.Array  # 0
    refund_of: jax.Array  # reserved: acquisition_id of the receipt this row credits
    cost: jax.Array  # F64[R]
    charged: jax.Array  # F64[R], reserved: what the receipt debited; equals cost
    payload: jax.Array  # F64[P]
    valid: jax.Array


def record_none(caps):
    """An invalid record (what a handler returns when it writes nothing)."""
    z = jnp.asarray(0, I64)
    return Record(
        **{n: z for n in RECORD_INT_FIELDS if n != "refund_of"},
        refund_of=jnp.asarray(-1, I64),
        cost=jnp.zeros(caps.n_resources, F64),
        charged=jnp.zeros(caps.n_resources, F64),
        payload=jnp.zeros(caps.payload_width, F64),
        valid=jnp.asarray(False),
    )


class RecordLog(eqx.Module):
    """The record log: the columns of `Record` with a leading row axis."""

    kind: jax.Array
    subject: jax.Array
    channel: jax.Array
    channel_index: jax.Array
    lane: jax.Array
    acquisition_index: jax.Array
    start_tick: jax.Array
    end_tick: jax.Array
    available_tick: jax.Array
    status: jax.Array
    phase: jax.Array
    acquisition_id: jax.Array
    revision: jax.Array
    released_tick: jax.Array
    accepted_tick: jax.Array
    superseded_by: jax.Array
    audience: jax.Array
    source: jax.Array
    vintage: jax.Array
    config_id: jax.Array
    refund_of: jax.Array
    cost: jax.Array  # F64[L, R]
    charged: jax.Array  # F64[L, R]
    payload: jax.Array  # F64[L, P]
    valid: jax.Array


def empty_log(caps):
    """Return an empty fixed-capacity record log."""
    z = jnp.zeros(caps.n_log, I64)
    return RecordLog(
        **{n: z for n in LOG_INT_FIELDS if n != "refund_of"},
        refund_of=jnp.full(caps.n_log, -1, I64),
        cost=jnp.zeros((caps.n_log, caps.n_resources), F64),
        charged=jnp.zeros((caps.n_log, caps.n_resources), F64),
        payload=jnp.zeros((caps.n_log, caps.payload_width), F64),
        valid=jnp.zeros(caps.n_log, B),
    )


def write_log(carry, rec, caps):
    """Append `rec` to the log where it is valid and a row is free.

    A full log sets `overflow` (bit `Capacity.LOG_ROWS`), skips the write and returns
    the clamped slot; the caller checks `n_log` before enqueueing a Release for the row.
    """
    ok = rec.valid & (carry.n_log < caps.n_log)
    s = jnp.minimum(carry.n_log, caps.n_log - 1)
    cols = {}
    for name in (*LOG_INT_FIELDS, "cost", "charged", "payload"):
        col = getattr(carry.log, name)
        cols[name] = col.at[s].set(jnp.where(ok, getattr(rec, name), col[s]))
    cols["valid"] = carry.log.valid.at[s].set(jnp.where(ok, True, carry.log.valid[s]))
    log = RecordLog(**cols)
    out = eqx.tree_at(
        lambda c: (c.log, c.n_log, c.overflow, c.overflow_mask),
        carry,
        (
            log,
            carry.n_log + ok.astype(I64),
            carry.overflow | (rec.valid & ~ok),
            carry.overflow_mask | jnp.where(rec.valid & ~ok, Capacity.LOG_ROWS, 0),
        ),
    )
    return out, s


def refit_rows(log, caps, audience, subject):
    """The rows a refit of `subject` for `audience` reads, as `n_log` log rows.

    Selection takes, for each acquisition of the subject, the latest revision the
    audience has received (accepted by the observer, released to the policy) first;
    that row counts only when its kind is in `caps.refit_kinds` and its status in
    `caps.refit_statuses`, so a correction with an unlisted status retracts the
    acquisition rather than leaving its earlier value. Counting rows come first in
    canonical order (channel, acquisition index, revision) with `valid` true; every
    other row follows with `valid` false. Rows are masked for the audience.
    """
    seen = log.accepted_tick if audience == Audience.OBSERVER else log.released_tick
    mine = log.valid & (seen >= 0) & (log.subject == subject)
    same = (log.acquisition_id[:, None] == log.acquisition_id[None, :]) & mine[None, :]
    later = same & (log.revision[None, :] > log.revision[:, None])
    newer = later.any(axis=1)
    # supersession as this audience sees it: a revision it has not received (an
    # observer-only correction, for the policy) must not mark an earlier row
    replaced = jnp.max(jnp.where(later, log.revision[None, :], -1), axis=1)
    kinds = jnp.asarray(caps.refit_kinds, I64)
    statuses = jnp.asarray(caps.refit_statuses, I64)
    counts = mine & ~newer & jnp.isin(log.kind, kinds) & jnp.isin(log.status, statuses)
    last = (~counts).astype(I64)
    order = jnp.lexsort((log.revision, log.acquisition_index, log.channel, last))
    # `RecordLog` holds the columns of `Record` in the same order, so the sorted
    # log's leaves are a batch of records
    picked = jax.tree_util.tree_map(
        lambda x: x[order],
        dataclasses.replace(log, valid=counts, superseded_by=replaced),
    )
    leaves = jax.tree_util.tree_leaves(picked)
    rows = jax.tree_util.tree_unflatten(
        jax.tree_util.tree_structure(record_none(caps)), leaves
    )
    return masked_record(rows, caps, audience)


RECORD_INT_FIELDS = tuple(
    f.name
    for f in dataclasses.fields(Record)
    if f.name not in ("cost", "charged", "payload", "valid")
)
LOG_INT_FIELDS = tuple(
    f.name
    for f in dataclasses.fields(RecordLog)
    if f.name not in ("cost", "charged", "payload", "valid")
)
