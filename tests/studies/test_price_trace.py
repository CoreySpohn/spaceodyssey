"""Host extraction and export of each world's written package price trace."""

import csv
import dataclasses
from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey.core.carry import empty_trace, init_carry
from spaceodyssey.core.enums import F64, I64, AdmissionCode, EffectKind
from spaceodyssey.core.records import pack
from spaceodyssey.core.transaction import admit, close_decision, finish_trace
from spaceodyssey.studies import bracket as br
from spaceodyssey.studies.effect_rows import effects
from spaceodyssey.studies.selection import trace_scalars
from tests.studies.bracket_helpers import build

POLICY = br.PackagePolicy(trace_capacity=5)
CAPS = SimpleNamespace(n_subjects=2)
LAYOUT = POLICY.state_layout(CAPS)
STUDY = SimpleNamespace(caps=CAPS, agenda=SimpleNamespace(policies=(POLICY,)))


def hand_policy_state(n_trace):
    """Distinct counters and poisoned unwritten rows in a codec-packed state."""
    trace = jnp.asarray(
        [
            [0.1, 2.0, 8.0, 10.0, 1.0, 0.2],
            [0.2, 5.0, 6.0, 9.0, 1.0, 0.5],
            [0.3, 5.0, 0.0, 7.0, 0.0, 0.5],
            [99.0] * 6,
            [99.0] * 6,
        ],
        F64,
    )
    state = dataclasses.replace(
        LAYOUT,
        last_tick=jnp.asarray([11, 17], I64),
        n_imaging=jnp.asarray([2, 3], I64),
        n_spectra=jnp.asarray([1, 0], I64),
        kick=jnp.asarray(2, I64),
        stopped=jnp.asarray(True),
        trace=trace,
        n_trace=jnp.asarray(n_trace, I64),
    )
    return jnp.pad(pack(state), (0, 4)), np.asarray(trace)


def price_trace_of(pstate, layout):
    """Read one world through the result API and its declared validity length."""
    assert layout.trace.shape == LAYOUT.trace.shape
    result = SimpleNamespace(carry=SimpleNamespace(pstate=pstate[None, :]))
    rows, lengths = br.price_trace(result, STUDY)
    return rows[0, : lengths[0]]


def run_grants_result(built):
    """Use the completed fixture run through the public grants runner."""

    class CompletedExecutor:
        def run(self, study, truths, roots, n_steps, **kwargs):
            return built.run

    return br.run_grants(
        {"oracle": (built.study, built.executor.boundary)},
        built.truths,
        built.roots,
        0,
        executor=CompletedExecutor(),
    )


def test_the_trace_is_read_from_the_packed_policy_state():
    """Fails if the trace is read by offset or past its written rows."""
    pstate, trace = hand_policy_state(n_trace=3)
    np.testing.assert_array_equal(price_trace_of(pstate, LAYOUT), trace[:3])


def test_world_lengths_exclude_unwritten_rows_from_csv(tmp_path):
    """Fails if unequal or empty world traces leak padding or overwrite an arm."""
    states = [hand_policy_state(n)[0] for n in (3, 1, 0)]
    result = SimpleNamespace(carry=SimpleNamespace(pstate=jnp.stack(states)))
    rows, lengths = br.price_trace(result, STUDY)
    np.testing.assert_array_equal(lengths, [3, 1, 0])
    for w, n in enumerate(lengths):
        np.testing.assert_array_equal(rows[w, :n], hand_policy_state(n)[1][:n])
        assert np.all(rows[w, n:] == 0)
    table = br.ActivityTable(
        star={"n_visits": np.zeros((3, 2))},
        world={"charged": np.zeros(3)},
        price_trace=(rows, lengths),
    )
    other_rows, other_lengths = br.price_trace(
        SimpleNamespace(carry=SimpleNamespace(pstate=states[1][None, :])), STUDY
    )
    other = br.ActivityTable(
        star={"n_visits": np.zeros((1, 2))},
        world={"charged": np.zeros(1)},
        price_trace=(other_rows, other_lengths),
    )
    br.ArmsResult(activity={"one": table, "two": other}).to_csv(tmp_path)
    with (tmp_path / "price_trace.csv").open(newline="") as handle:
        exported = list(csv.DictReader(handle))
    assert [(r["arm"], int(r["world"]), int(r["row"])) for r in exported] == [
        ("one", 0, 0),
        ("one", 0, 1),
        ("one", 0, 2),
        ("one", 1, 0),
        ("two", 0, 0),
    ]
    columns = ("price", "cum_charged_d", "selected_d", "window_d", "acted", "cum_value")
    np.testing.assert_array_equal(
        [[float(row[name]) for name in columns] for row in exported],
        np.concatenate([rows[0, :3], rows[1, :1], other_rows[0, :1]]),
    )


@pytest.fixture(scope="module")
def built():
    return build(horizon_d=30.0, budget_d=200.0)


@eqx.filter_jit
def admit_one(carry, effect, study):
    """Record one real admission without running the physical completion model."""
    carry, codes = admit(carry, effect, study)
    carry = close_decision(carry, effect, carry.pstate, True, study)
    trace = finish_trace(
        dataclasses.replace(empty_trace(study.caps), wake=jnp.asarray(True)),
        effect,
        codes,
        jnp.asarray(0.0, F64),
        True,
        carry,
        study,
    )
    return carry, trace


@pytest.mark.slow(reason="shares the chunked physical run and compiles admission once")
def test_spending_first_packages_counts_every_later_funds_refusal(built):
    """Fails if package refusals are hidden or kickoff, waits or padding count."""
    study = built.study
    resource = study.channel_model.mode_table.resource("telescope_days")
    cost_d = study.channels.cost[0, resource]
    carries, streams, states = [], [], []
    for w, funded in enumerate((2, 8)):
        carry = init_carry(study, None)
        carry = dataclasses.replace(
            carry,
            ledger=dataclasses.replace(
                carry.ledger,
                balance=carry.ledger.balance.at[-1, resource].set(funded * cost_d),
            ),
        )
        stream = []
        for i in range(study.caps.n_subjects + 4):
            channel = (
                study.channel_model.mode_table.catalog_channel
                if i < study.caps.n_subjects
                else 0
            )
            carry = dataclasses.replace(
                carry,
                tick=jnp.asarray(i, I64) * (study.channels.occupancy_ticks[0] + 1),
            )
            carry, trace = admit_one(
                carry,
                effects(study.caps, [EffectKind.OBSERVE], channel_index=channel),
                study,
            )
            stream.append(trace)
        codes = np.asarray([t.admit_code[0] for t in stream])
        np.testing.assert_array_equal(codes[: study.caps.n_subjects], AdmissionCode.OK)
        np.testing.assert_array_equal(
            codes[study.caps.n_subjects :],
            [AdmissionCode.OK] * 2
            + [AdmissionCode.NO_FUNDS if w == 0 else AdmissionCode.OK] * 2,
        )
        np.testing.assert_array_max_ulp(
            np.asarray(carry.ledger.balance[-1, resource]),
            np.asarray((funded - min(funded, 4)) * cost_d),
            maxulp=4,
        )
        # These refusal codes must be ignored because they are not packages.
        stream[0] = dataclasses.replace(
            stream[0],
            admit_code=jnp.full_like(stream[0].admit_code, AdmissionCode.NO_FUNDS),
        )
        stream.append(
            dataclasses.replace(
                stream[-1],
                eff_kind=jnp.full_like(trace.eff_kind, EffectKind.WAIT),
                admit_code=jnp.full_like(trace.admit_code, AdmissionCode.NO_FUNDS),
            )
        )
        stream.append(
            dataclasses.replace(
                stream[-2],
                wake=jnp.asarray(False),
                admit_code=jnp.full_like(trace.admit_code, AdmissionCode.NO_FUNDS),
            )
        )
        carries.append(carry)
        streams.append(jax.tree.map(lambda *xs: jnp.stack(xs), *stream))
        layout = study.agenda.policies[0].state_layout(study.caps)
        state = dataclasses.replace(
            layout,
            trace=jnp.full_like(layout.trace, 99.0),
            n_trace=jnp.asarray(1 - w, I64),
        )
        states.append(jnp.pad(pack(state), (0, study.caps.n_pstate - pack(state).size)))
    result = SimpleNamespace(
        carry=dataclasses.replace(
            built.run.carry,
            ledger=jax.tree.map(
                lambda *xs: jnp.stack(xs), *(c.ledger for c in carries)
            ),
            pstate=jnp.stack(states),
        ),
        trace=jax.tree.map(lambda *xs: jnp.stack(xs), *streams),
    )
    table = br.activity_table(result, study)
    np.testing.assert_array_equal(table.world["early_overspend"], [2, 0])
    np.testing.assert_array_equal(table.world["stranded_d"], [0.0, 0.0])


@pytest.mark.slow(reason="compiles the chunked refresh driver once")
def test_every_run_writes_its_trace_and_scalars(built, tmp_path):
    """Fails if a result leaves out the trace or scalars, or counts unwritten rows."""
    table = br.activity_table(built.run, built.study)
    rows, lengths = br.price_trace(built.run, built.study)
    resource = built.study.channel_model.mode_table.resource("telescope_days")
    for w, n in enumerate(lengths):
        trace = rows[w, :n]
        balance_d = built.run.carry.ledger.balance[w, -1, resource]
        over, stranded = trace_scalars(
            jnp.asarray(trace),
            final_balance_d=balance_d,
            admission_code=built.run.trace.admit_code[w].ravel(),
            admission_late=(
                built.run.trace.wake[w, :, None]
                & (built.run.trace.n_dec[w, :, None] > built.study.caps.n_subjects)
                & (built.run.trace.eff_kind[w] == EffectKind.OBSERVE)
            ).ravel(),
        )
        assert table.world["early_overspend"][w] == over == 0
        assert table.world["stranded_d"][w] == stranded
    assert np.any(rows[:, :, 3] > 0)
    assert np.all(table.world["stranded_d"] > 0)
    result = run_grants_result(built)
    np.testing.assert_array_equal(result.activity["oracle"].price_trace[1], lengths)
    result.to_csv(tmp_path)
    with (tmp_path / "price_trace.csv").open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == int(lengths.sum())
    assert (tmp_path / "grants.txt").exists()
