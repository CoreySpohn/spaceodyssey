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
    """Integer codes for audience."""

    OBSERVER = 1
    POLICY = 2


PUBLIC = Audience.OBSERVER | Audience.POLICY


def effect_mask(*kinds):
    """Bitmask over `EffectKind` with one bit per listed kind."""
    mask = 0
    for kind in kinds:
        mask |= 1 << int(kind)
    return mask


ALL_EFFECTS = effect_mask(*EffectKind)
