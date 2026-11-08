"""A process that writes failure receipts, and the toy study that prices and refunds."""

import dataclasses
import inspect
import textwrap

import equinox as eqx
import jax
import jax.numpy as jnp

from spaceodyssey.core import AbstractProcess, EventKind, Stage, enqueue, record_none
from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.declarations import Declaration
from spaceodyssey.core.enums import F64, I64, PUBLIC, ChargePricing, RecordStatus
from spaceodyssey.studies import toy
from tests.core.program_helpers import ScriptedPrograms

RECEIPT_CHANNEL = 50
FAILURE_KIND = toy.RECORD_AUXILIARY  # a policy-visible kind of the toy registry
BIG = 10_000
BALANCE = 100.0


N_RECEIPTS = 3  # every study carries this many rows, so the studies share a compile
OFF = (0, 0, 10, 0, 0, 0, 3, 0)


def receipt(tick, *, subject=0, channel=10, index=0, delay=0, held=False, audience=3):
    """A failure receipt fired at `tick`, released `delay` ticks later or held."""
    return (tick, subject, channel, index, delay, int(held), audience, 1)


def mutate(fn, old, new):
    """`fn` with the source text `old` replaced by `new`, in the module of `fn`."""
    source = textwrap.dedent(inspect.getsource(fn))
    assert old in source, old
    scope = dict(fn.__globals__)
    exec(source.replace(old, new, 1), scope)
    return scope[fn.__name__]


class ScriptedReceipts(AbstractProcess):
    """Writes one failure receipt per scripted row.

    A row is (tick, subject, target channel id, target acquisition index, delay, held,
    audience, on); a row that is not `on` fires nothing. The receipt names its target
    by that semantic identity only: the target channel is its `source`, the index its
    `refund_index`. It is available `delay` ticks after it is written, and a held one
    is never released.
    """

    rows: jax.Array = eqx.field(converter=lambda x: jnp.asarray(x, I64))

    def declare(self):
        """The record channel this process writes on, for pre-flight."""
        return Declaration(channels=(RECEIPT_CHANNEL,))

    def seed(self, carry, caps):
        """Queue one process event per receipt, at its tick and to its audience."""
        for i in range(self.rows.shape[0]):
            row = self.rows[i]
            carry = enqueue(
                carry,
                row[7] == 1,
                kind=EventKind.PROCESS,
                tick=row[0],
                stage=Stage.RELEASE,
                group=i,
                audience=row[6],
                caps=caps,
            )
        return carry

    def handle(self, carry, ev, study):
        """Write the receipt of this fire."""
        table = self.rows[ev.group]
        no_refunds = not study.caps.refunds
        status = jnp.where(
            table[5] == 1, RecordStatus.UNRELEASED, RecordStatus.COMPLETED
        )
        rec = dataclasses.replace(
            record_none(study.caps),
            kind=FAILURE_KIND if no_refunds else study.channels.failure_kind[0],
            subject=table[1],
            channel=jnp.asarray(RECEIPT_CHANNEL, I64),
            source=table[2],
            refund_index=None if no_refunds else table[3],
            start_tick=ev.tick,
            end_tick=ev.tick,
            available_tick=ev.tick + table[4],
            status=status.astype(I64),
            acquisition_id=carry.next_sid,
            released_tick=jnp.asarray(-1, I64),
            accepted_tick=jnp.asarray(-1, I64),
            superseded_by=jnp.asarray(-1, I64),
            valid=jnp.asarray(True),
        )
        carry = eqx.tree_at(lambda c: c.next_sid, carry, carry.next_sid + 1)
        return carry, rec


def _column(value, c, r, dtype):
    return jnp.broadcast_to(jnp.asarray(value, dtype), (c, r))


def refund_study(
    rows=(),
    receipts=(),
    *,
    charge=None,
    pricing=None,
    surcharge=None,
    deadline=BIG,
    failure=FAILURE_KIND,
    refunds=True,
    cap_kw=None,
    policy=None,
    **overrides,
):
    """The toy study with refunds on, a scripted policy and scripted receipts.

    `charge`, `pricing` and `surcharge` fill channel 10 (index 0) across the three
    resources; the other channels keep their default. Every balance row holds
    `BALANCE`, so a debit never refuses.
    """
    caps = Capacities(n_effects=3, refunds=refunds, **(cap_kw or {}))
    base = toy.survey_1(caps=caps)
    c, r = caps.n_channels, caps.n_resources
    cost = jnp.full((c, r), 10.0, F64)
    ch = dataclasses.replace(base.channels, cost=cost, charge=cost)
    if charge is not None:
        ch = dataclasses.replace(ch, charge=ch.charge.at[0].set(charge))
    if refunds:
        # Absent columns change the pytree structure, so a study that sets only some of
        # them compiles a program of its own. Every refund study carries all of them,
        # filled with the values an absent column stands for, and shares one program.
        pricing = ChargePricing.OCCUPANCY if pricing is None else pricing
        surcharge = 0.0 if surcharge is None else surcharge
        table = jnp.full((c, r), ChargePricing.OCCUPANCY, I64)
        ch = dataclasses.replace(
            ch,
            pricing=table.at[0].set(pricing),
            surcharge=jnp.zeros((c, r), F64).at[0].set(surcharge),
        )
    if failure is not None:
        kinds = jnp.full(c, -1, I64).at[0].set(failure)
        ch = dataclasses.replace(
            ch,
            failure_kind=kinds,
            refund_deadline_ticks=jnp.full(c, deadline, I64),
        )
    budget = jnp.full((caps.n_balance_rows, r), BALANCE, F64)
    fires = (*receipts, *[OFF] * (N_RECEIPTS - len(receipts)))
    process = ScriptedReceipts(rows=fires)
    return toy.survey_1(
        caps=caps,
        channels=ch,
        budget=budget,
        policies=(policy or ScriptedPrograms(script=rows),),
        process=process,
        **overrides,
    )


__all__ = ["PUBLIC", "ScriptedReceipts", "mutate", "receipt", "refund_study"]
