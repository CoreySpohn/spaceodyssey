"""Capacities."""

import dataclasses
import math

import jax

from spaceodyssey.core.enums import BIG, N_STAGES, N_TIES
from spaceodyssey.core.events import _order_key_py

# Field metadata that marks a declaration field as optional in a document: the
# encoder omits it at its default and the loader fills it when missing. Only fields
# added after the first document format carry it, so every older document keeps its
# hash.
OPTIONAL = {"optional": True}

PEDIGREE_HEAD = 3  # audience, refit status, row count
PEDIGREE_TRIPLE = 3  # channel, acquisition index, revision


def pedigree_width(rows_per_subject):
    """Columns of a pedigree that names up to `rows_per_subject` counting rows.

    Zero when the refresh protocol is off (`rows_per_subject == 0`), so a Slice 1
    study carries a zero-size product column.
    """
    if rows_per_subject <= 0:
        return 0
    return PEDIGREE_HEAD + PEDIGREE_TRIPLE * rows_per_subject


@dataclasses.dataclass(frozen=True)
class Capacities:
    """Static capacities; a different `Capacities` retraces by design."""

    n_subjects: int = 4  # the corpus uses subjects 0 and 1; n_an solves need four
    n_assets: int = 2
    n_lanes: int = 3
    n_programs: int = 1
    n_resources: int = 3  # r0 asset-scoped, r1 program-scoped, r2 shared
    n_channels: int = 3
    n_events: int = 48
    n_log: int = 40
    payload_width: int = 4  # a floor: `KindRegistry.lower` widens it to the widest kind
    # Audience flags per record kind and payload column, lowered from the registry.
    # Empty means no registry was lowered and every column is readable.
    payload_audience: tuple[tuple[int, ...], ...] = ()
    # Registered kind names by index and the indices of derived kinds, lowered from
    # the registry; and the observer's refit kind indices and statuses (empty when it
    # does not refit), lowered by `KindRegistry.lower`.
    # `kind_names` is read only while a study is built or checked, never under trace, so
    # it stays out of the equality and hash that key a compiled program.
    kind_names: tuple[str, ...] = dataclasses.field(default=(), compare=False)
    derived_kinds: tuple[int, ...] = ()
    refit_kinds: tuple[int, ...] = ()
    refit_statuses: tuple[int, ...] = ()
    # The refresh protocol: counting rows a refit reads per subject (0 disables the
    # protocol structurally) and the world lanes of one masked refit call.
    rows_per_subject: int = 0
    refit_lanes: int = 0
    # Columns of the pedigree on a product row (`RecordLog.product`): audience, refit
    # status and row count, then one (channel, acquisition index, revision) triple per
    # counting row. Zero without the refresh protocol; with it, zero means the width
    # `pedigree_width(rows_per_subject)`, which is the least a protocol study needs.
    product_width: int = 0
    n_effects: int = 2
    n_pstate: int = 4
    n_an: int = 4
    max_micro: int = 8
    micro_guard: int = 6
    max_sid: int = 4096
    # Slice 3a features. Each is off at its default, and a leaf that exists only for
    # a feature has zero size (and leaves every history and fingerprint as they were)
    # while the feature is off. They are optional document fields (`OPTIONAL`).
    awards: bool = dataclasses.field(default=False, metadata=OPTIONAL)
    refunds: bool = dataclasses.field(default=False, metadata=OPTIONAL)
    n_commitments: int = dataclasses.field(default=0, metadata=OPTIONAL)
    snapshot_balances: bool = dataclasses.field(default=False, metadata=OPTIONAL)
    # The unit the policy's integer balance view counts in; read only while
    # `snapshot_balances` is on.
    balance_quantum: float = dataclasses.field(default=1.0, metadata=OPTIONAL)

    @property
    def n_balance_rows(self) -> int:
        """Number of asset, program, and shared balance rows."""
        return self.n_assets + self.n_programs + 1

    @property
    def max_tick(self) -> int:
        """Largest tick representable below the invalid-row sentinel."""
        return BIG // (N_STAGES * self.max_micro * N_TIES * self.max_sid) - 1

    def __post_init__(self):
        """Validate integer arithmetic bounds and static shapes."""
        require_x64()
        if not self.micro_guard < self.max_micro:
            raise ValueError("micro_guard must be below max_micro")
        if not self.max_tick > 0:
            raise ValueError("max_tick must be positive")
        if any(len(row) != self.payload_width for row in self.payload_audience):
            raise ValueError("every payload_audience row needs payload_width entries")
        if self.rows_per_subject < 0 or self.refit_lanes < 0 or self.product_width < 0:
            raise ValueError(
                "rows_per_subject, refit_lanes and product_width must not be negative"
            )
        need = pedigree_width(self.rows_per_subject)
        if self.rows_per_subject > 0 and self.product_width == 0:
            object.__setattr__(self, "product_width", need)
        if self.product_width < need:
            raise ValueError(
                f"product_width {self.product_width} is below the pedigree width "
                f"{need} that rows_per_subject {self.rows_per_subject} needs"
            )
        n_commit = self.n_commitments
        if isinstance(n_commit, bool) or not isinstance(n_commit, int) or n_commit < 0:
            raise ValueError("n_commitments must be a nonnegative integer")
        if not isinstance(self.awards, bool):
            raise ValueError("awards must be a bool")
        if not isinstance(self.refunds, bool):
            raise ValueError("refunds must be a bool")
        if not isinstance(self.snapshot_balances, bool):
            raise ValueError("snapshot_balances must be a bool")
        quantum = self.balance_quantum
        if (
            isinstance(quantum, bool)
            or not isinstance(quantum, float)
            or not math.isfinite(quantum)
            or quantum <= 0.0
        ):
            raise ValueError("balance_quantum must be a positive finite float")
        if not self.n_an <= self.n_subjects:
            raise ValueError("n_an must not exceed n_subjects")
        largest = _order_key_py(
            self.max_tick,
            N_STAGES - 1,
            self.max_micro - 1,
            self.max_sid - 1,
            self,
            N_TIES - 1,
        )
        if not largest < BIG:
            raise ValueError("largest order key reaches BIG")


def require_x64():
    """Require 64-bit tick arithmetic without changing global JAX settings."""
    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError(
            "spaceodyssey.core requires JAX x64; enable it at startup with "
            'jax.config.update("jax_enable_x64", True).'
        )
