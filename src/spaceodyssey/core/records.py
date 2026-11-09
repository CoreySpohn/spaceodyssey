"""Records."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp

from spaceodyssey.core.capacities import (
    OPTIONAL,
    PEDIGREE_HEAD,
    PEDIGREE_TRIPLE,
    Capacities,
)
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


def hide_refund_of(refund_of):
    """`refund_of` as a view shows it: the legacy -1 where a row names no origin.

    The column holds -1 on every legacy row, so a value of zero or more is an internal
    acquisition id a refund resolved. It is queue-derived and never shown.
    """
    return jnp.where(refund_of >= 0, -1, refund_of)


def masked_record(rec, caps, audience):
    """Hide queue-derived identities and payload columns the audience cannot read."""
    keep = payload_columns(caps, rec.kind, audience)
    out = dataclasses.replace(
        rec,
        acquisition_id=jnp.zeros_like(rec.acquisition_id),
        refund_of=hide_refund_of(rec.refund_of) if caps.refunds else rec.refund_of,
        payload=jnp.where(keep, rec.payload, 0.0),
    )
    if rec.program is not None and audience != Audience.POLICY:
        # a program id is the policy's to read; the observer is shown none
        out = dataclasses.replace(out, program=jnp.zeros_like(rec.program))
    return out


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
    # The program the acquisition was charged under; absent (None, no leaf) unless the
    # capacities hold more than one program, so a study with one is unchanged.
    program: jax.Array | None = None
    # Refund columns, absent (None, no leaf) unless the capacities hold refunds: the
    # fixed part of the charge, the semantic acquisition index a failure receipt names
    # (visible; `refund_of` holds the internal id the handler resolves it to), and the
    # window generation of the payer row at debit.
    nonrefundable: jax.Array | None = dataclasses.field(
        default=None, metadata=OPTIONAL
    )  # F64[R]
    refund_index: jax.Array | None = dataclasses.field(default=None, metadata=OPTIONAL)
    award_gen: jax.Array | None = dataclasses.field(default=None, metadata=OPTIONAL)
    # Under awards: the window generation, at debit, of the row that paid each resource
    # (one entry per payer slot: asset, program, shared), I64[R].
    award_gens: jax.Array | None = dataclasses.field(default=None, metadata=OPTIONAL)
    # and the remainder rule and reallocation destination of that row's window at the
    # debit (the rule is NONE for a debit made outside a window), I64[R] each.
    award_rules: jax.Array | None = dataclasses.field(default=None, metadata=OPTIONAL)
    award_tos: jax.Array | None = dataclasses.field(default=None, metadata=OPTIONAL)
    # The bundle the acquisition was admitted in (`decision * max_bundles + ordinal`, -1
    # for none); present only while bundles are on, and visible to every audience.
    bundle: jax.Array | None = dataclasses.field(default=None, metadata=OPTIONAL)
    # The ops state of the asset when the acquisition was admitted, which priced it;
    # present only while ops states are on.
    ops: jax.Array | None = dataclasses.field(default=None, metadata=OPTIONAL)


def record_none(caps):
    """An invalid record (what a handler returns when it writes nothing)."""
    z = jnp.asarray(0, I64)
    return Record(
        **{n: z for n in RECORD_INT_FIELDS if n != "refund_of"},
        award_gens=jnp.zeros(caps.n_resources, I64) if caps.awards else None,
        award_rules=jnp.zeros(caps.n_resources, I64) if caps.awards else None,
        award_tos=jnp.zeros(caps.n_resources, I64) if caps.awards else None,
        refund_of=jnp.asarray(-1, I64),
        cost=jnp.zeros(caps.n_resources, F64),
        charged=jnp.zeros(caps.n_resources, F64),
        payload=jnp.zeros(caps.payload_width, F64),
        valid=jnp.asarray(False),
        program=z if caps.n_programs > 1 else None,
        bundle=jnp.asarray(-1, I64) if caps.bundles else None,
        ops=z if caps.ops else None,
        **(
            dict(
                nonrefundable=jnp.zeros(caps.n_resources, F64),
                refund_index=z,
                award_gen=z,
            )
            if caps.refunds
            else {}
        ),
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
    program: jax.Array | None  # I64[L], present only with more than one program
    # the refund columns of `Record`, present only while refunds are on
    nonrefundable: jax.Array | None  # F64[L, R]
    refund_index: jax.Array | None
    award_gen: jax.Array | None
    award_gens: jax.Array | None  # I64[L, R], present only with awards
    award_rules: jax.Array | None  # I64[L, R], present only with awards
    award_tos: jax.Array | None  # I64[L, R], present only with awards
    bundle: jax.Array | None  # I64[L], present only with bundles
    ops: jax.Array | None  # I64[L], present only with ops states
    # Pedigree of a product row, I64[L, product_width]: audience, refit status, row
    # count and the revision set as (channel, acquisition index, revision) triples. It
    # is not a `Record` field: only a refresh writes it; every other row holds zeros.
    product: jax.Array


def log_records(log, **replace):
    """The rows of `log` as a `Record` batch (the pedigree is not a record field).

    Keyword arguments replace the named columns.
    """
    # `RecordLog` lists the fields of `Record` in order, then `product` last
    rows = Record(**{f.name: vars(log)[f.name] for f in dataclasses.fields(Record)})
    return dataclasses.replace(rows, **replace)


def log_columns(log):
    """The columns a log holds, by name (no program column with one program)."""
    return {n: v for n, v in log.__dict__.items() if v is not None}


REFUND_COLUMNS = ("nonrefundable", "refund_index", "award_gen")
AWARD_COLUMNS = ("award_gens", "award_rules", "award_tos")
BUNDLE_COLUMNS = ("bundle",)
OPS_COLUMNS = ("ops",)


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
        program=z if caps.n_programs > 1 else None,
        nonrefundable=jnp.zeros((caps.n_log, caps.n_resources), F64)
        if caps.refunds
        else None,
        refund_index=z if caps.refunds else None,
        award_gen=z if caps.refunds else None,
        product=jnp.zeros((caps.n_log, caps.product_width), I64),
        award_gens=jnp.zeros((caps.n_log, caps.n_resources), I64)
        if caps.awards
        else None,
        award_rules=jnp.zeros((caps.n_log, caps.n_resources), I64)
        if caps.awards
        else None,
        award_tos=jnp.zeros((caps.n_log, caps.n_resources), I64)
        if caps.awards
        else None,
        bundle=jnp.full(caps.n_log, -1, I64) if caps.bundles else None,
        ops=z if caps.ops else None,
    )


def write_log(carry, rec, caps):
    """Append `rec` to the log where it is valid and a row is free.

    A full log sets `overflow` (bit `Capacity.LOG_ROWS`), skips the write and returns
    the clamped slot; the caller checks `n_log` before enqueueing a Release for the row.

    A row that reuses the acquisition id of a logged row is a correction, and a
    correction may not involve a derived kind: a derived row never corrects or is
    corrected. Such a row is refused (not logged, no release) and sets `overflow` with
    the `Capacity.INVALID_CORRECTION` bit, a refusal that is not a capacity.
    """
    bad = jnp.asarray(False)
    if caps.derived_kinds:
        old = carry.log.valid & (carry.log.acquisition_id == rec.acquisition_id)
        mine = derived_rows(rec, caps)
        bad = rec.valid & (old & (derived_rows(carry.log, caps) | mine)).any()
    ok = rec.valid & ~bad & (carry.n_log < caps.n_log)
    s = jnp.minimum(carry.n_log, caps.n_log - 1)
    cols = {}
    for name in (*LOG_INT_FIELDS, "cost", "charged", "payload"):
        col = getattr(carry.log, name)
        cols[name] = col.at[s].set(jnp.where(ok, getattr(rec, name), col[s]))
    cols["valid"] = carry.log.valid.at[s].set(jnp.where(ok, True, carry.log.valid[s]))
    col = carry.log.program
    if col is not None:
        col = col.at[s].set(jnp.where(ok, rec.program, col[s]))
    for name in (*REFUND_COLUMNS, *AWARD_COLUMNS, *BUNDLE_COLUMNS, *OPS_COLUMNS):
        old = getattr(carry.log, name)
        cols[name] = (
            None
            if old is None
            else old.at[s].set(jnp.where(ok, getattr(rec, name), old[s]))
        )
    log = RecordLog(**cols, program=col, product=carry.log.product)
    out = eqx.tree_at(
        lambda c: (c.log, c.n_log, c.overflow, c.overflow_mask),
        carry,
        (
            log,
            carry.n_log + ok.astype(I64),
            carry.overflow | (rec.valid & ~ok),
            carry.overflow_mask
            | jnp.where(bad, Capacity.INVALID_CORRECTION, 0)
            | jnp.where(rec.valid & ~ok & ~bad, Capacity.LOG_ROWS, 0),
        ),
    )
    return out, s


def derived_rows(log, caps):
    """Boolean per log row: its kind is derived (a product or a declared summary)."""
    return jnp.isin(log.kind, jnp.asarray(caps.derived_kinds, I64))


def revision_set(log, caps, audience, subject):
    """`(counts, replaced)` over the log rows: the revision set of a subject.

    For each acquisition of the subject, the latest revision the audience has received
    (accepted by the observer, released to the policy) replaces every earlier one: that
    is the replacement rule, and `replaced` is, per row, the revision that replaced it
    as this audience sees it (-1 if none). A revision this audience has not received (an
    observer-only correction, for the policy) replaces nothing. `counts` marks the rows
    of the set: the current revision of each acquisition, when its kind is in
    `caps.refit_kinds` and its status in `caps.refit_statuses`, so a correction with an
    unlisted status retracts the acquisition rather than leaving its earlier value.
    A derived kind is never in `refit_kinds`, so a product is never in a set.
    """
    seen = log.accepted_tick if audience == Audience.OBSERVER else log.released_tick
    # a derived row neither replaces nor is replaced: it has its own internal id, and a
    # correction that reuses one never touches a source row
    mine = log.valid & (seen >= 0) & (log.subject == subject)
    mine = mine & ~derived_rows(log, caps)
    same = (log.acquisition_id[:, None] == log.acquisition_id[None, :]) & mine[None, :]
    later = same & (log.revision[None, :] > log.revision[:, None])
    newer = later.any(axis=1)
    replaced = jnp.max(jnp.where(later, log.revision[None, :], -1), axis=1)
    kinds = jnp.asarray(caps.refit_kinds, I64)
    statuses = jnp.asarray(caps.refit_statuses, I64)
    counts = mine & ~newer & jnp.isin(log.kind, kinds) & jnp.isin(log.status, statuses)
    return counts, replaced


def refit_rows(log, caps, audience, subject):
    """The rows a refit of `subject` for `audience` reads, as `n_log` log rows.

    The rows of the revision set (`revision_set`) come first in canonical order
    (channel, acquisition index, revision) with `valid` true; every other row follows
    with `valid` false. Rows are masked for the audience.
    """
    counts, replaced = revision_set(log, caps, audience, subject)
    last = (~counts).astype(I64)
    order = jnp.lexsort((log.revision, log.acquisition_index, log.channel, last))
    rows = log_records(
        jax.tree_util.tree_map(lambda x: x[order], log),
        valid=counts[order],
        superseded_by=replaced[order],
    )
    return masked_record(rows, caps, audience)


def pedigree_of(log, slot):
    """`(audience, status, triples)` of the product row at `slot`, read from the log.

    `triples` is the revision set as `(channel, acquisition index, revision)` tuples in
    canonical order. Meaningful only for a row a refresh wrote.
    """
    row = [int(x) for x in jnp.asarray(log.product)[slot]]
    audience, status, n = row[:PEDIGREE_HEAD]
    flat = row[PEDIGREE_HEAD : PEDIGREE_HEAD + PEDIGREE_TRIPLE * n]
    triples = tuple(
        tuple(flat[i : i + PEDIGREE_TRIPLE])
        for i in range(0, len(flat), PEDIGREE_TRIPLE)
    )
    return audience, status, triples


_NOT_INT = (
    "cost",
    "charged",
    "payload",
    "valid",
    "product",
    "program",
    *REFUND_COLUMNS,
    *AWARD_COLUMNS,
    *BUNDLE_COLUMNS,
    *OPS_COLUMNS,
)
RECORD_INT_FIELDS = tuple(
    f.name for f in dataclasses.fields(Record) if f.name not in _NOT_INT
)
LOG_INT_FIELDS = tuple(
    f.name for f in dataclasses.fields(RecordLog) if f.name not in _NOT_INT
)
