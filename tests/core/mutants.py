import types

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jax import lax

from spaceodyssey import core
from spaceodyssey.core import handlers, transaction
from spaceodyssey.core.executors.compiled import CompiledExecutor
from spaceodyssey.core.executors.host import HostExecutor


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
)


def _mutant_step(carry, ev, study, truth, root):
    return _step_with_table(carry, ev, study, truth, root, MUTANT_HANDLERS)


def _padded_step(carry, ev, study, truth, root):
    return _step_with_table(
        carry, ev, study, truth, root, handlers.HANDLERS + (handlers.h_none,) * 4
    )


_mutant_advance = _clone(transaction.advance, step=_mutant_step)
_mutant_transition = _clone(transaction.transition, advance=_mutant_advance)
_mutant_advance_jit = eqx.filter_jit(_mutant_advance)
_padded_advance = _clone(transaction.advance, step=_padded_step)
_padded_transition = _clone(transaction.transition, advance=_padded_advance)
_padded_advance_jit = eqx.filter_jit(_padded_advance)


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
