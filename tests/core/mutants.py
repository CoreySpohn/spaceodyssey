import types

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jax import lax

from spaceodyssey import core
from spaceodyssey.core import handlers, transaction
from spaceodyssey.core.executors.compiled import CompiledExecutor
from spaceodyssey.core.executors.host import HostExecutor
from spaceodyssey.core.keys import fold64


def _clone(fn, **replacements):
    return types.FunctionType(
        fn.__code__,
        dict(fn.__globals__, **replacements),
        fn.__name__,
        fn.__defaults__,
        fn.__closure__,
    )


def _mutant_realize(c, study, truth, root, base, fraction):
    def draw(root, coords, purpose):
        return jr.fold_in(core.draw_key(root, coords, purpose), c.n_dec)

    return _clone(handlers._realize_rec, draw_key=draw)(
        c, study, truth, root, base, fraction
    )


def _step_with_table(carry, ev, study, truth, root, table):
    branches = tuple((lambda c, h=h: h(c, ev, study, truth, root)) for h in table)
    return lax.switch(
        jnp.clip(ev.kind, 0, len(table) - 1).astype(jnp.int32), branches, carry
    )


_cancel = _clone(handlers._cancel_lane, _realize_rec=_mutant_realize)
MUTANT_HANDLERS = (
    handlers.h_none,
    _clone(handlers.h_done, _realize_rec=_mutant_realize),
    handlers.h_release,
    handlers.h_wait,
    _clone(handlers.h_stop, _cancel_lane=_cancel),
    handlers.h_process,
    _clone(handlers.h_cancel, _cancel_lane=_cancel),
    handlers.h_none,
    _clone(handlers.h_fault, _cancel_lane=_cancel),
)


def _mutant_step(carry, ev, study, truth, root):
    return _step_with_table(carry, ev, study, truth, root, MUTANT_HANDLERS)


def _padded_step(carry, ev, study, truth, root):
    return _step_with_table(
        carry, ev, study, truth, root, handlers.HANDLERS + (handlers.h_none,) * 3
    )


_mutant_advance = _clone(transaction.advance, step=_mutant_step)
_mutant_transition = _clone(transaction.transition, advance=_mutant_advance)
_mutant_advance_jit = eqx.filter_jit(_mutant_advance)
_padded_advance = _clone(transaction.advance, step=_padded_step)
_padded_transition = _clone(transaction.transition, advance=_padded_advance)
_padded_advance_jit = eqx.filter_jit(_padded_advance)


def _count_decision_key(carry, root, study):
    """The mutant policy key: the decision count, as before keys followed meaning."""
    return fold64(fold64(root, core.Purpose.POLICY), carry.n_dec)


_count_decide = _clone(transaction.decide, decision_key=_count_decision_key)
_count_transition = _clone(transaction.transition, decide=_count_decide)
_count_decide_jit = eqx.filter_jit(_count_decide)


class CountKeyCompiledExecutor(CompiledExecutor):
    def transition(
        self, carry, i, root, study, truth, *, wake_fn=core.wake_by_audience
    ):
        return _count_transition(carry, i, root, study, truth, wake_fn=wake_fn)


class CountKeyHostExecutor(HostExecutor):
    def decide(self, carry, root, study):
        return _count_decide_jit(carry, root, study)


class MutantCompiledExecutor(CompiledExecutor):
    def transition(
        self, carry, i, root, study, truth, *, wake_fn=core.wake_by_audience
    ):
        return _mutant_transition(carry, i, root, study, truth, wake_fn=wake_fn)


class MutantHostExecutor(HostExecutor):
    def advance(self, carry, root, study, truth, *, wake_fn=core.wake_by_audience):
        return _mutant_advance_jit(carry, root, study, truth, wake_fn=wake_fn)


class PaddedCompiledExecutor(CompiledExecutor):
    def transition(
        self, carry, i, root, study, truth, *, wake_fn=core.wake_by_audience
    ):
        return _padded_transition(carry, i, root, study, truth, wake_fn=wake_fn)


class PaddedHostExecutor(HostExecutor):
    def advance(self, carry, root, study, truth, *, wake_fn=core.wake_by_audience):
        return _padded_advance_jit(carry, root, study, truth, wake_fn=wake_fn)


def _never_waits(carry, caps):
    """The per-wake rule: a decision is due after every waking event."""
    return jnp.asarray(True)


_per_wake_advance = _clone(transaction.advance, decision_due=_never_waits)
_per_wake_advance_jit = eqx.filter_jit(_per_wake_advance)


class PerWakeHostExecutor(HostExecutor):
    """Negative control: one decision per waking event instead of one per tick."""

    def advance(self, carry, root, study, truth, *, wake_fn=core.wake_by_audience):
        return _per_wake_advance_jit(carry, root, study, truth, wake_fn=wake_fn)


def _h_cancel_decrementing(c, ev, study, truth, root):
    """Negative control: a cancel frees the acquisition index it consumed."""
    c1, emit, rec = handlers.h_cancel(c, ev, study, truth, root)
    counts = c1.acquisition_counts.at[rec.subject].add(-emit.n_written)
    return eqx.tree_at(lambda x: x.acquisition_counts, c1, counts), emit, rec


DECREMENTING_HANDLERS = (
    *handlers.HANDLERS[:6],
    _h_cancel_decrementing,
    *handlers.HANDLERS[7:],
)


def _decrementing_step(carry, ev, study, truth, root):
    return _step_with_table(carry, ev, study, truth, root, DECREMENTING_HANDLERS)


_decrementing_advance = _clone(transaction.advance, step=_decrementing_step)
decrementing_advance_jit = eqx.filter_jit(_decrementing_advance)
