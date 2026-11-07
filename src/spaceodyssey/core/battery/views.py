"""Capture and compare the masked views actually handed to policies."""

import dataclasses
import itertools
import weakref

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from spaceodyssey.core.battery.result import bits_equal, tree_bits_equal
from spaceodyssey.core.enums import Audience, EventKind
from spaceodyssey.core.keys import policy_key
from spaceodyssey.core.kinds import AbstractPolicy
from spaceodyssey.core.records import log_columns, payload_columns
from spaceodyssey.core.snapshot import snapshot_fingerprint, visible_rows

# Log columns that hold a time, named rather than recognized by sign: a datum observed
# before the run starts has a negative start and end tick.
TICKS = ("start_tick", "end_tick", "available_tick", "released_tick", "accepted_tick")
# Stamps that hold -1 until their event happens. No other tick column has an unset
# value, so -1 there is a time like any other.
STAMP_TICKS = ("released_tick", "accepted_tick")
RELEASE_TICKS = {"available_tick", "released_tick", "accepted_tick"}


def moved_ticks(name, column, valid, shift):
    """Log column `name` with its times on `valid` rows moved by `shift` ticks.

    A column outside `TICKS` is returned as it is. A tick column moves on every valid
    row, whatever its sign; a release or acceptance stamp still at -1 (unset) stays.
    """
    column = np.asarray(column)
    if name not in TICKS:
        return column
    moving = np.asarray(valid, bool)
    if name in STAMP_TICKS:
        moving = moving & (column != -1)
    return np.where(moving, column + shift, column)


_CAPTURES = weakref.WeakValueDictionary()
_CAPTURE_IDS = itertools.count()


def _deliver(slot, key, phase, *args):
    """Hand a recorded decision to the capture that owns `slot`."""
    _CAPTURES[int(slot)]._record(key, phase, *args)


class _RecordingPolicy(AbstractPolicy):
    """Record what the inner policy sees.

    The capture is named by a traced `slot`, never held as a static callable, so every
    capture of the same study shares one compiled program.
    """

    inner: AbstractPolicy
    slot: jax.Array
    phase: int = eqx.field(static=True)

    def decide(self, pstate, view, study, key):
        effects, state, draw = self.inner.decide(pstate, view, study, key)
        fingerprint = snapshot_fingerprint(view, study.observer.layout(study.caps))
        leaves, layout = jax.tree_util.tree_flatten((fingerprint, view, effects))
        specs = [(x.shape, np.dtype(x.dtype)) for x in leaves]
        # Callback workers can use 32-bit mode even when the caller enables x64.
        # Transport 64-bit values as words so their bytes survive device placement.
        words = [
            jax.lax.bitcast_convert_type(x, jnp.uint32) if x.dtype.itemsize == 8 else x
            for x in leaves
        ]

        def record(slot, key, *values):
            restored = [
                np.asarray(x).view(dtype).reshape(shape)
                for x, (shape, dtype) in zip(values, specs, strict=True)
            ]
            _deliver(slot, key, phase, *jax.tree_util.tree_unflatten(layout, restored))

        phase = self.phase
        jax.debug.callback(record, self.slot, key, *words)
        return effects, state, draw

    def decide_py(self, pstate, view_np, study, key):
        effects, state, draw = self.inner.decide_py(pstate, view_np, study, key)
        # Host-form policies run eagerly on the exact view the executor built.
        fingerprint = snapshot_fingerprint(view_np, study.observer.layout(study.caps))
        _deliver(self.slot, key, self.phase, fingerprint, view_np, effects)
        return effects, state, draw


class DecisionViews:
    """Per-run capture, indexed by decision key, ordinal and snapshot identity.

    Compiled transitions can evaluate inactive decisions or phase branches. Only
    entries named by handled, woken trace rows are retained by `read`. Fingerprints
    select the executed view here; relations compare its columns, never its digest.
    """

    def __init__(self):
        """Start an empty capture for one run."""
        self.entries = {}
        self._slot = next(_CAPTURE_IDS)
        _CAPTURES[self._slot] = self

    def _record(self, key, phase, fingerprint, view, effects):
        address = (
            *np.asarray(key).tolist(),
            int(phase),
            int(view.tick),
            int(view.n_dec),
            int(fingerprint),
        )
        self.entries[address] = jax.tree_util.tree_map(np.asarray, (view, effects))

    def instrument(self, study):
        """Wrap policies to record their inputs without changing outputs."""
        policies = tuple(
            _RecordingPolicy(inner=p, slot=jnp.asarray(self._slot, jnp.int32), phase=i)
            for i, p in enumerate(study.agenda.policies)
        )
        return dataclasses.replace(
            study, agenda=dataclasses.replace(study.agenda, policies=policies)
        )

    def read(self, study, root, trace):
        """Return the captured view and effects for each executed decision."""
        jax.effects_barrier()
        rows = np.flatnonzero(np.asarray(trace.handled) & np.asarray(trace.wake))
        out = []
        previous, ordinal = None, 0
        for i in rows:
            tick = int(trace.tick[i])
            ordinal = ordinal + 1 if previous == tick else 0
            previous = tick
            phase = int((tick >= np.asarray(study.agenda.phase_until[:-1])).sum())
            key = np.asarray(policy_key(root, phase, tick, ordinal)).tolist()
            n_dec = int(trace.n_dec[i])
            fp = int(trace.snapshot_fp[i])
            # A decision that wakes on the step that ends the run is not counted
            # (its commit is gated off), so its view still carries the earlier count.
            found = self.entries.get((*key, phase, tick, n_dec - 1, fp))
            if found is None:
                found = self.entries.get((*key, phase, tick, n_dec, fp))
            out.append(found)
        return out


def _payload_matches_log(view, carry, caps):
    """Draws may change under id relabeling; each view must still show its own draw.

    A view row is matched among the log rows the policy can see, by identity and
    audience, so the observer's copy of a datum written once per audience is never a
    candidate. Two visible rows that match one view row are refused (`ValueError`).
    """
    log = carry.log
    names = ("subject", "acquisition_index", "channel", "revision", "status")
    names += ("audience",)
    seen = np.asarray(visible_rows(log, Audience.POLICY))
    for i in range(int(view.n_rows)):
        hit = seen.copy()
        for name in names:
            hit &= np.asarray(getattr(log, name)) == getattr(view.log, name)[i]
        rows = np.flatnonzero(hit)
        if len(rows) > 1:
            raise ValueError(
                f"log rows {rows.tolist()} share the identity of policy view row {i}"
            )
        if len(rows) != 1:
            return False
        j = rows[0]
        want = np.where(
            payload_columns(caps, log.kind[j], Audience.POLICY), log.payload[j], 0.0
        )
        if not bits_equal(view.log.payload[i], want):
            return False
    return not np.any(np.asarray(view.log.payload)[int(view.n_rows) :])


def compare_views(
    left, right, ca, cb, caps, trace, *, layout, channel_map=None, epoch=0, delay=0
):
    """Compare decision views under an id, epoch or release-time transformation.

    `layout` is the observer's `BeliefLayout`; under a relabeling only the summary's
    count leaf is compared, since draws move with the new ids. The `left` view's tick
    columns move by `epoch`, and its release ticks also by `delay`, before comparison
    (`moved_ticks`); payload columns never move.
    """
    if len(left) != len(right):
        return ["the number of policy decision views changed"]
    failures = []
    steps = np.flatnonzero(np.asarray(trace.handled) & np.asarray(trace.wake))
    for ordinal, (a, b) in enumerate(zip(left, right, strict=True)):
        if a is None or b is None:
            failures.append(f"decision {ordinal}: the policy view was not captured")
            continue
        va, ea = a
        vb, eb = b
        prefix = f"decision {ordinal}: policy view"
        for name in ("n_rows", "n_dec", "release_version", "tick"):
            x = np.asarray(getattr(va, name))
            if name == "tick":
                x = (
                    x
                    + epoch
                    + (
                        delay
                        if int(trace.kind[steps[ordinal]]) == EventKind.RELEASE
                        else 0
                    )
                )
            if not bits_equal(x, getattr(vb, name)):
                failures.append(f"{prefix} {name} changed")
        for name, col in log_columns(va.log).items():
            x = np.asarray(col)
            valid = np.asarray(va.log.valid)
            if channel_map and name in ("channel", "source"):
                mapped = np.vectorize(
                    lambda v: channel_map.get(int(v), int(v)), otypes=[x.dtype]
                )(x)
                x = np.where(valid, mapped, x)
            shift = epoch + (delay if name in RELEASE_TICKS else 0)
            x = moved_ticks(name, x, valid, shift)
            if channel_map and name == "payload":
                if not (
                    _payload_matches_log(va, ca, caps)
                    and _payload_matches_log(vb, cb, caps)
                ):
                    failures.append(f"{prefix} payload differs from its masked records")
                continue
            if not bits_equal(x, getattr(vb.log, name)):
                failures.append(f"{prefix} log {name} changed")
        if channel_map:
            if not bits_equal(layout.count(va.belief), layout.count(vb.belief)):
                failures.append(f"{prefix} belief counts changed")
        elif not tree_bits_equal(va.belief, vb.belief):
            failures.append(f"{prefix} belief changed")
        if delay and not tree_bits_equal(ea, eb):
            failures.append(
                f"decision {ordinal}: requested effects changed under the delay"
            )
    return failures
