"""Capacities."""

import dataclasses

import jax

from spaceodyssey.core.enums import BIG, N_STAGES, N_TIES
from spaceodyssey.core.events import _order_key_py


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
    n_effects: int = 2
    n_pstate: int = 4
    n_an: int = 4
    max_micro: int = 8
    micro_guard: int = 6
    max_sid: int = 4096

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
    if not jax.config.x64_enabled:
        raise RuntimeError(
            "spaceodyssey.core requires JAX x64; enable it at startup with "
            'jax.config.update("jax_enable_x64", True).'
        )
