"""Capture and compare the masked views actually handed to policies."""

import dataclasses
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from spaceodyssey.core.battery.result import bits_equal, tree_bits_equal
from spaceodyssey.core.enums import Audience, EventKind
from spaceodyssey.core.keys import policy_key
from spaceodyssey.core.kinds import AbstractPolicy
from spaceodyssey.core.records import payload_columns
from spaceodyssey.core.snapshot import snapshot_fingerprint

TICKS = {"start_tick", "end_tick", "available_tick", "released_tick", "accepted_tick"}
RELEASE_TICKS = {"available_tick", "released_tick", "accepted_tick"}


class _RecordingPolicy(AbstractPolicy):
    inner: AbstractPolicy
    record: Callable = eqx.field(static=True)
    phase: int = eqx.field(static=True)

    def decide(self, pstate, view, study, key):
        effects, state, draw = self.inner.decide(pstate, view, study, key)
        leaves, layout = jax.tree_util.tree_flatten(
            (snapshot_fingerprint(view), view, effects)
        )
        specs = [(x.shape, np.dtype(x.dtype)) for x in leaves]
        # Callback workers can use 32-bit mode even when the caller enables x64.
        # Transport 64-bit values as words so their bytes survive device placement.
        words = [
            jax.lax.bitcast_convert_type(x, jnp.uint32) if x.dtype.itemsize == 8 else x
            for x in leaves
        ]

        def record(key, *values):
            restored = [
                np.asarray(x).view(dtype).reshape(shape)
                for x, (shape, dtype) in zip(values, specs, strict=True)
            ]
            self.record(
                key, self.phase, *jax.tree_util.tree_unflatten(layout, restored)
            )

        jax.debug.callback(record, key, *words)
        return effects, state, draw

    def decide_py(self, pstate, view_np, study, key):
        effects, state, draw = self.inner.decide_py(pstate, view_np, study, key)
        # Host-form policies run eagerly on the exact view the executor built.
        self.record(key, self.phase, snapshot_fingerprint(view_np), view_np, effects)
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
            _RecordingPolicy(inner=p, record=self._record, phase=i)
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
    """Draws may change under id relabeling; each view must still show its own draw."""
    log = carry.log
    names = ("subject", "acquisition_index", "channel", "revision", "status")
    for i in range(int(view.n_rows)):
        hit = np.asarray(log.valid).copy()
        for name in names:
            hit &= np.asarray(getattr(log, name)) == getattr(view.log, name)[i]
        rows = np.flatnonzero(hit)
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
    left, right, ca, cb, caps, trace, *, channel_map=None, epoch=0, delay=0
):
    """Compare decision views under an id, epoch or release-time transformation."""
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
        for name in type(va.log).__dataclass_fields__:
            x = np.asarray(getattr(va.log, name))
            valid = np.asarray(va.log.valid)
            if channel_map and name in ("channel", "source"):
                mapped = np.vectorize(
                    lambda v: channel_map.get(int(v), int(v)), otypes=[x.dtype]
                )(x)
                x = np.where(valid, mapped, x)
            if name in TICKS:
                shift = epoch + (delay if name in RELEASE_TICKS else 0)
                x = np.where(valid & (x >= 0), x + shift, x)
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
            if not bits_equal(va.belief.n, vb.belief.n):
                failures.append(f"{prefix} belief counts changed")
        elif not tree_bits_equal(va.belief, vb.belief):
            failures.append(f"{prefix} belief changed")
        if delay and not tree_bits_equal(ea, eb):
            failures.append(
                f"decision {ordinal}: requested effects changed under the delay"
            )
    return failures
