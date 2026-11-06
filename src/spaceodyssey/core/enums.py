"""Enums."""

from enum import IntEnum, IntFlag

import jax.numpy as jnp
import numpy as np

I64 = jnp.int64
F64 = jnp.float64
B = jnp.bool_
N_STAGES = 5
N_TIES = 3
TIE_NEUTRAL = 1
BIG = int(np.iinfo(np.int64).max // 4)


class Stage(IntEnum):
    """Integer codes for stage."""

    SETTLE = 0
    RELEASE = 1
    ANALYZE = 2
    DECIDE = 3
    ADMIT = 4


class EventKind(IntEnum):
    """Integer codes for event kinds."""

    NONE = 0
    DONE = 1
    RELEASE = 2
    WAIT = 3
    STOP = 4
    PROCESS = 5
    CANCEL = 6
    RESULT = 7
    FAULT = 8
    REQUEST = 9


class EffectKind(IntEnum):
    """Integer codes for effect kinds."""

    NONE = 0
    OBSERVE = 1
    WAIT = 2
    STOP = 3
    CANCEL = 4
    ALLOCATE = 5
    QUERY = 6


class Scope(IntEnum):
    """Integer codes for scope."""

    LANE = 0
    ASSET = 1
    PROGRAM = 2
    CAMPAIGN = 3


class Disposition(IntEnum):
    """Integer codes for disposition."""

    CLOSE = 0
    DRAIN = 1
    ABORT = 2


class RecordStatus(IntEnum):
    """Integer codes for record statuses."""

    COMPLETED = 0
    FAILED = 1
    PARTIAL = 2
    CANCELLED = 3
    NULL = 4  # the acquisition ran and produced no usable value
    UNRELEASED = 5  # logged but held back: never released, never folded


class Purpose(IntEnum):
    """Integer codes for purpose."""

    WORLD = 0
    MEASUREMENT = 1
    POLICY = 2
    INFERENCE = 3
    PROCESS = 4


class AdmissionCode(IntEnum):
    """Integer codes for admission results."""

    OK = 0
    NO_FUNDS = 1
    LANE_BUSY = 2
    LANE_CLOSED = 3
    NO_SLOT = 4
    HORIZON = 5
    UNSUPPORTED = 6
    NONE = 7
    OUTSIDE_VALIDITY = 8
    FROZEN_HORIZON = 9
    FORBIDDEN = 10


class CoincidentOrder(IntEnum):
    """How a channel orders a fault against a settlement at the same tick and stage."""

    SETTLE_FIRST = 0
    FAULT_FIRST = 1


class ChargeRule(IntEnum):
    """How a program row is charged (reserved; only the receipt rule exists)."""

    RECEIPT = 0


class RemainderRule(IntEnum):
    """What happens to an unspent balance at the end of its window (reserved)."""

    NONE = 0
    FORFEIT = 1
    CARRY_OVER = 2
    REALLOCATE = 3


class Audience(IntFlag):
    """Who may read a record or a payload column.

    A record's audience says who it is released to (the observer always folds what it
    is released; the policy sees it only with the `POLICY` flag). A payload column's
    audience says who may read it: `SCORER` alone marks a column that stays in the log
    for scoring and never reaches a snapshot or the observer.
    """

    OBSERVER = 1
    POLICY = 2
    SCORER = 4


class Capacity(IntFlag):
    """Which static capacity ran out; the bits of `Carry.overflow_mask`."""

    QUEUE_SLOTS = 1
    STABLE_IDS = 2
    LOG_ROWS = 4


class RunStatus(IntEnum):
    """How a run ended; only `COMPLETE` means the study ran to its own end.

    `STALLED` is the microstep guard: a row reached the guard position, so the run
    stopped instead of looping within one tick. `OVERFLOW` names no capacity by
    itself; `Carry.overflow_mask` holds the `Capacity` bits. `AWAITING` means a
    request to the host is still unanswered, so nothing after it ran.
    """

    COMPLETE = 0
    IDLE = 1
    TRUNCATED = 2
    STALLED = 3
    OVERFLOW = 4
    AWAITING = 5


def version_slot(audience):
    """Index of an audience in `Carry.release_version`: 0 observer, 1 policy."""
    return 1 if int(audience) == int(Audience.POLICY) else 0


PUBLIC = Audience.OBSERVER | Audience.POLICY
EVERYONE = Audience.OBSERVER | Audience.POLICY | Audience.SCORER


def effect_mask(*kinds):
    """Bitmask over `EffectKind` with one bit per listed kind."""
    mask = 0
    for kind in kinds:
        mask |= 1 << int(kind)
    return mask


ALL_EFFECTS = effect_mask(*EffectKind)

# Effects each Agenda level permits, by level number. The levels are nested: each
# mask holds every bit of the one before it.
_LEVEL_0 = effect_mask(EffectKind.NONE, EffectKind.CANCEL)
_LEVEL_1 = _LEVEL_0 | effect_mask(
    EffectKind.OBSERVE, EffectKind.WAIT, EffectKind.QUERY, EffectKind.STOP
)
_LEVEL_3 = _LEVEL_1 | effect_mask(EffectKind.ALLOCATE)
LEVEL_EFFECTS = (_LEVEL_0, _LEVEL_1, _LEVEL_1, _LEVEL_3)
