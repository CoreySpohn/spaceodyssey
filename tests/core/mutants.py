import types

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import lax

from spaceodyssey import core
from spaceodyssey.core import Audience, handlers, transaction
from spaceodyssey.core.enums import RecordStatus
from spaceodyssey.core.executors.compiled import CompiledExecutor
from spaceodyssey.core.executors.host import HostExecutor
from spaceodyssey.core.executors.replay import _fold, _row
from spaceodyssey.core.keys import fold64
from spaceodyssey.core.records import masked_record


def _clone(fn, **replacements):
    return types.FunctionType(
        fn.__code__,
        dict(fn.__globals__, **replacements),
        fn.__name__,
        fn.__defaults__,
        fn.__closure__,
    )


def _mutant_realize(c, study, truth, root, base, fraction):
    def draw(root, coords, purpose, **kw):
        return jr.fold_in(core.draw_key(root, coords, purpose, **kw), c.n_dec)

    return _clone(handlers._realize_rec, draw_key=draw)(
        c, study, truth, root, base, fraction
    )


def _step_with_table(carry, ev, study, truth, root, table):
    # the real step leaves out the handlers of a feature that is off
    left_out = (() if study.caps.awards else (handlers.h_window_end,)) + (
        () if study.caps.commitments else (handlers.h_commit_start,)
    )
    table = tuple(handlers.h_none if h in left_out else h for h in table)
    branches = tuple((lambda c, h=h: h(c, ev, study, truth, root)) for h in table)
    return lax.switch(
        jnp.clip(ev.kind, 0, len(table) - 1).astype(jnp.int32), branches, carry
    )


_cancel = _clone(
    handlers._cancel_lane,
    _cancel_one=_clone(handlers._cancel_one, _realize_rec=_mutant_realize),
)
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


def _count_fold_release(c, ev, study, truth, root):
    """The mutant fold key: the record's key folded with the decision count."""

    def key(root, rec, study):
        return jr.fold_in(core.fold_key(root, rec, study), c.n_dec)

    return _clone(handlers.h_release, fold_key=key)(c, ev, study, truth, root)


COUNT_FOLD_HANDLERS = tuple(
    {
        2: _count_fold_release,
        7: _clone(handlers.h_result, h_release=_count_fold_release),
    }.get(i, h)
    for i, h in enumerate(handlers.HANDLERS)
)


def _count_fold_step(carry, ev, study, truth, root):
    return _step_with_table(carry, ev, study, truth, root, COUNT_FOLD_HANDLERS)


_count_fold_advance = _clone(transaction.advance, step=_count_fold_step)
_count_fold_transition = _clone(transaction.transition, advance=_count_fold_advance)
_count_fold_advance_jit = eqx.filter_jit(_count_fold_advance)


class CountFoldKeyCompiledExecutor(CompiledExecutor):
    def transition(
        self, carry, i, root, study, truth, *, wake_fn=core.wake_by_audience
    ):
        return _count_fold_transition(carry, i, root, study, truth, wake_fn=wake_fn)


class CountFoldKeyHostExecutor(HostExecutor):
    def advance(self, carry, root, study, truth, *, wake_fn=core.wake_by_audience):
        return _count_fold_advance_jit(carry, root, study, truth, wake_fn=wake_fn)


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


def live_belief_replay(study, carry, trace):
    """A replay that returns the live belief: it passes any equality check."""
    return carry.belief


def initial_belief_replay(study, carry, trace):
    """A replay that folds nothing."""
    return study.observer.init(study.caps)


def log_order_replay(study, carry, trace):
    """A replay that folds the first logged revision instead of the first released."""
    log = carry.log
    valid = np.asarray(log.valid)
    status = np.asarray(log.status)
    audience = np.asarray(log.audience)
    acquisition = np.asarray(log.acquisition_id)
    belief = study.observer.init(study.caps)
    folded = set()
    for i in range(len(valid)):
        if not valid[i] or status[i] == RecordStatus.UNRELEASED:
            continue
        if not int(audience[i]) & Audience.OBSERVER or int(acquisition[i]) in folded:
            continue
        folded.add(int(acquisition[i]))
        row = masked_record(_row(log, i), study.caps, Audience.OBSERVER)
        belief = _fold(study, belief, row)
    return belief
