"""Lower planit plan tables on the host and replay their due rows."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.constants import d2s
from planit_py import PlanTable

from spaceodyssey.core.capacities import require_x64
from spaceodyssey.core.enums import EffectKind
from spaceodyssey.core.kinds import AbstractPolicy
from spaceodyssey.core.snapshot import EffectTable
from spaceodyssey.core.study import Clock, Study


def plan_table_to_ticks(table: PlanTable, clock: Clock) -> np.ndarray:
    """Convert valid start days to int64 ticks, once, without relative tolerance.

    Padding returns zero ticks. The absolute tolerance in ticks is
    ``min(1e-3, max(1e-9, 4 * ulp(x)))``. A study horizon is checked by
    :meth:`PlanTablePolicy.from_plan_table`, since a Clock holds only tick length.
    """
    require_x64()
    length = float(clock.tick_length_s)
    if not np.isfinite(length) or length <= 0:
        raise ValueError("tick_length_s must be finite and positive")
    days = np.asarray(table.start_times_d, dtype=np.float64)
    valid = np.asarray(table.valid)
    ticks = np.zeros(days.shape, dtype=np.int64)
    for row in np.flatnonzero(valid):
        day = days[row]
        if not np.isfinite(day) or day < 0:
            raise ValueError(f"row {row}: day {day!r} is non-finite or negative")
        with np.errstate(over="ignore", invalid="ignore"):
            x = day * d2s / length
        rounded = np.rint(x)
        if not np.isfinite(x) or rounded >= 2**63:
            raise ValueError(f"row {row}: day {day!r} exceeds int64 ticks")
        offset = x - rounded
        tolerance = min(1e-3, max(1e-9, 4 * np.spacing(x)))
        if abs(offset) > tolerance:
            raise ValueError(
                f"row {row}: day {day!r} has offset {offset!r} ticks "
                f"from the nearest tick (tolerance {tolerance!r})"
            )
        ticks[row] = int(rounded)
    return ticks


class PlanTablePolicy(AbstractPolicy):
    """Replay the next valid row when its start tick is due at a decision point.

    ``pstate[0]`` is the next row cursor; a due row consumes it once, regardless
    of admission. Every selected channel is emitted in one atomic effect group.
    Channel ids become study channel indices at load time, and each row is padded
    to ``caps.n_effects`` slots. Plan values are array leaves, so changing values
    with the same shapes reuses the executor programs.

    The study owns decision wakes and channel timing, including transitions. This
    adapter supports default transition kind 0 and checks that the row asset owns
    every selected channel lane. Groups spanning several plan rows are refused:
    one decision replays one row, so it cannot guarantee atomic admission across
    rows. Before a row is due and after exhaustion, the policy emits no effects.
    """

    subject: jax.Array
    channel_mask: jax.Array
    channel_index: jax.Array
    start_tick: jax.Array
    valid: jax.Array
    group: jax.Array
    version: int = eqx.field(default=1, kw_only=True)

    @classmethod
    def from_plan_table(cls, table: PlanTable, study: Study) -> "PlanTablePolicy":
        """Validate and lower a concrete plan table for this study on the host."""
        ticks = plan_table_to_ticks(table, study.clock)
        valid = np.asarray(table.valid, dtype=bool)
        subjects = np.asarray(table.target_ids)
        ids = np.asarray(table.channel_ids)
        study_ids = np.asarray(study.channels.id)
        row_masks = np.asarray(table.channel_mask)
        groups = np.asarray(table.group_ids)
        channel_index = np.zeros((len(valid), study.caps.n_effects), dtype=np.int64)
        channel_mask = np.zeros_like(channel_index, dtype=bool)
        if study.caps.n_pstate < 1:
            raise ValueError("PlanTablePolicy needs at least one pstate slot")
        for cid in ids:
            if cid not in study_ids:
                raise ValueError(f"unknown channel {cid}")
        for row in np.flatnonzero(valid):
            if subjects[row] < 0 or subjects[row] >= study.caps.n_subjects:
                raise ValueError(f"row {row}: unknown subject {subjects[row]}")
            if ticks[row] > int(study.horizon_tick):
                raise ValueError(
                    f"row {row}: day {table.start_times_d[row]} beyond horizon"
                )
            if int(table.transition_kinds[row]) != 0:
                raise ValueError(f"row {row}: unsupported transition kind")
            if groups[row] >= 0 and np.sum(valid & (groups == groups[row])) > 1:
                raise ValueError(f"row {row}: group spans multiple plan rows")
            selected = np.flatnonzero(np.isin(study_ids, ids[row_masks[row]]))
            if len(selected) == 0 or len(selected) > study.caps.n_effects:
                raise ValueError(f"row {row}: channel set does not fit effect capacity")
            lanes = np.asarray(study.channels.lane)[selected]
            assets = np.asarray(study.assets.lane_asset)[lanes]
            if np.any(assets != int(table.asset_ids[row])):
                raise ValueError(
                    f"row {row}: asset does not own selected channel lanes"
                )
            channel_index[row, : len(selected)] = selected
            channel_mask[row, : len(selected)] = True
        return cls(
            subject=jnp.asarray(np.where(valid, subjects, 0), jnp.int64),
            channel_mask=jnp.asarray(channel_mask),
            channel_index=jnp.asarray(channel_index),
            start_tick=jnp.asarray(ticks),
            valid=jnp.asarray(valid),
            group=jnp.asarray(groups, jnp.int64),
        )

    def decide(self, pstate, view, study, key):
        """Emit one due row with fixed shapes and no traced Python branches."""
        return self._decide(pstate, view, study, jnp)

    def decide_py(self, pstate, view_np, study, key):
        """Replay the same row and cursor rule with NumPy arrays on the host."""
        return self._decide(pstate, view_np, study, np)

    def _decide(self, pstate, view, study, xp):
        """Apply the replay rule with the chosen numerical array operations."""
        valid = xp.asarray(self.valid)
        remaining = valid & (xp.arange(valid.shape[0]) >= pstate[0])
        row = xp.argmax(remaining)
        due = xp.any(remaining) & (xp.asarray(self.start_tick)[row] <= view.tick)
        mask = xp.asarray(self.channel_mask)[row] & due
        n = study.caps.n_effects
        zeros = xp.zeros(n, dtype=xp.int64)
        effects = EffectTable(
            kind=xp.where(mask, int(EffectKind.OBSERVE), int(EffectKind.NONE)),
            subject=xp.full(n, xp.asarray(self.subject)[row], dtype=xp.int64),
            channel_index=xp.asarray(self.channel_index)[row],
            group=xp.full(n, xp.asarray(self.group)[row], dtype=xp.int64),
            tick=zeros,
            lane=zeros,
            scope=zeros,
            disposition=zeros,
            disruptive=xp.zeros(n, dtype=bool),
        )
        state = xp.where(
            (xp.arange(len(pstate)) == 0) & due, row + 1, xp.asarray(pstate)
        )
        return effects, state, xp.asarray(0.0, dtype=xp.float64)
