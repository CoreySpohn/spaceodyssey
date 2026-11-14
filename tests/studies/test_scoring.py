"""Scorer joins and host folds on explicit rows, plus activity accounting."""

import dataclasses
from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis.strategies import permutations

from spaceodyssey.adapters.measurement import MeasurementRow, measurement_template
from spaceodyssey.adapters.modes import ModeTable
from spaceodyssey.adapters.parameters import catalog_template
from spaceodyssey.core.carry import empty_trace, init_carry
from spaceodyssey.core.enums import (
    I64,
    AdmissionCode,
    Audience,
    EffectKind,
    RecordStatus,
)
from spaceodyssey.core.kinds import AbstractPolicy
from spaceodyssey.core.records import pack, record_none, unpack, write_log
from spaceodyssey.core.snapshot import snapshot_view
from spaceodyssey.core.transaction import admit, finish_trace, mask_effects
from spaceodyssey.criteria.claims import AbstractClaim
from spaceodyssey.criteria.transition import TransitionTable
from spaceodyssey.studies import bracket as br
from spaceodyssey.studies.effect_rows import effects
from spaceodyssey.studies.scoring import (
    DUPLICATE,
    MISASSOCIATED,
    MISSED,
    CriterionObjective,
    refold,
    row_log,
    verdict,
)

from .bracket_helpers import N_SLOTS, build, fixture_modes

MODES = ModeTable(fixture_modes(), (1.0,))


class SmallRadiusClaim(AbstractClaim):
    """An explicit test predicate separating two catalog classes."""

    name: str = "small"

    def evaluate(self, params, luminosity_lsun):
        return params.radius_rearth < 2.0


CRITERION = TransitionTable(
    step_names=("spectrum",),
    labels=("met",),
    mode_names=(MODES.modes[1].name,),
    claims=(SmallRadiusClaim(),),
    target_claim="small",
    step_modes=[[True]],
    requirement_snr=[5.0],
    margin=[0.0],
    repeat=[1],
    on_pass=[1],
    on_fail=[-1],
    label_values=[0.7],
    start=0,
    step_labels=[-1],
)


@pytest.fixture(scope="module")
def base_study():
    study = build(run=False).study
    return dataclasses.replace(
        study, observer=dataclasses.replace(study.observer, criterion=CRITERION)
    )


def hand_scored_log(base=None, *, mis_associated=1, duplicate=1, missed=1):
    """Two rows: tracks 0->2, 1->0, 2->0, and a mixed track 3."""
    study = base if base is not None else build(run=False).study
    study = dataclasses.replace(
        study, observer=dataclasses.replace(study.observer, criterion=CRITERION)
    )
    carry = init_carry(study, None)
    catalog = jax.tree.map(
        lambda x: jnp.ones(x.shape, x.dtype), catalog_template(N_SLOTS, 2)
    )
    catalog = dataclasses.replace(
        catalog,
        present=jnp.asarray([True, True, True, False, False, False]),
        params=dataclasses.replace(
            catalog.params,
            ln_period_d=jnp.full(N_SLOTS, jnp.log(100.0)),
            ecc=jnp.zeros(N_SLOTS),
        ),
    )
    row = jax.tree.map(
        lambda x: jnp.zeros(x.shape, x.dtype), measurement_template(N_SLOTS, 2)
    )
    row = dataclasses.replace(
        row,
        mode=jnp.asarray(1, I64),
        valid_science=jnp.asarray(1.0),
        occupied_s=jnp.asarray(br.SECONDS_PER_DAY),
        assignment=jnp.asarray([0, 1, 2 if duplicate else -1, 3, -1, -1]),
        identity=jnp.asarray([2, 0, 0, 1, -1, -1]),
        counts=row.counts.at[:4, 1].set(6.0),
        variance=row.variance.at[:4, 1].set(1.0),
        true_counts=row.true_counts.at[:4, 1].set(6.0),
        true_variance=row.true_variance.at[:4, 1].set(1.0),
    )
    second = dataclasses.replace(
        row, identity=row.identity.at[3].set(2 if mis_associated else 1)
    )
    if not missed:
        catalog = dataclasses.replace(catalog, present=catalog.present.at[1].set(False))
    entries = [
        (br.KIND_CATALOG, catalog),
        (br.KIND_MEASUREMENT, row),
        (br.KIND_MEASUREMENT, second),
    ]
    for i, (kind, value) in enumerate(entries):
        payload = pack(value)
        rec = dataclasses.replace(
            record_none(study.caps),
            valid=jnp.asarray(True),
            kind=jnp.asarray(kind, I64),
            status=jnp.asarray(RecordStatus.COMPLETED, I64),
            subject=jnp.asarray(0, I64),
            end_tick=jnp.asarray(i + 1, I64),
            channel_index=jnp.asarray(1, I64),
            payload=jnp.pad(payload, (0, study.caps.payload_width - payload.size)),
        )
        carry, _ = write_log(carry, rec, study.caps)
        if kind == br.KIND_MEASUREMENT:
            carry = dataclasses.replace(
                carry, belief=study.observer.fold(carry.belief, rec, study.caps)
            )
    carry = dataclasses.replace(carry, tick=jnp.asarray(3, I64))
    return SimpleNamespace(
        carry=jax.tree.map(lambda x: x[None], carry), trace=None
    ), study


@pytest.fixture(scope="module")
def scored_case(base_study):
    return hand_scored_log(base_study)


@pytest.fixture(scope="module")
def folded_case(scored_case):
    result, study = scored_case
    rows = dict(row_log(result, study))
    rows["counts"] = rows["counts"] * 0.25
    state = CRITERION.init_state(N_SLOTS, 2)
    for i in range(len(rows["world"])):
        row = MeasurementRow(
            **{
                f.name: jnp.asarray(rows[f.name][i])
                for f in dataclasses.fields(MeasurementRow)
            }
        )
        state = CRITERION.update(state, row, MODES)
    return state, rows


def permute_detections(result, order):
    """Carry every detection leaf, including assignment and identity, together."""
    log = result.carry.log
    payload = log.payload
    for w, r in zip(
        *np.nonzero(
            np.asarray(log.valid) & (np.asarray(log.kind) == br.KIND_MEASUREMENT)
        ),
        strict=True,
    ):
        row = unpack(measurement_template(N_SLOTS, 2), payload[w, r])
        row = jax.tree.map(
            lambda x, order=order: x[jnp.asarray(order)] if x.ndim else x, row
        )
        packed = pack(row)
        payload = payload.at[w, r, : packed.size].set(packed)
    return SimpleNamespace(
        carry=dataclasses.replace(
            result.carry, log=dataclasses.replace(log, payload=payload)
        ),
        trace=result.trace,
    )


def test_a_refold_equals_the_incremental_fold(folded_case):
    """Fails if host and incremental folds disagree, or batching changes a sum."""
    inc, rows = folded_case
    np.testing.assert_allclose(
        refold(CRITERION, rows, 0, 0, MODES).sum_counts, inc.sum_counts, rtol=1e-12
    )
    np.testing.assert_array_equal(
        refold(CRITERION, rows, 0, 0, MODES, columns="true").node[:4], 1
    )
    np.testing.assert_array_equal(inc.node, 0)
    reversed_rows = {k: v[::-1] for k, v in rows.items()}
    np.testing.assert_array_equal(
        refold(CRITERION, reversed_rows, 0, 0, MODES).sum_counts, inc.sum_counts
    )
    assert rows["channel"].tolist() == [1, 1]
    assert rows["end_tick"].tolist() == [2, 3]
    empty = {k: v[:0] for k, v in rows.items()}
    np.testing.assert_array_equal(refold(CRITERION, empty, 0, 0, MODES).node, 0)


@settings(max_examples=25, deadline=None)
@given(order=permutations(range(6)))
def test_the_verdict_survives_a_permuted_detection_order(order, scored_case):
    """Fails if detection order changes a verdict joined by assignment and identity."""
    result, study = scored_case
    a, b = verdict(result, study), verdict(permute_detections(result, order), study)
    np.testing.assert_array_equal(a.scorer_label, b.scorer_label)
    np.testing.assert_array_equal(a.mission_label, b.mission_label)
    assert all(np.array_equal(a.counts[k], b.counts[k]) for k in a.counts)
    np.testing.assert_array_equal(a.yield_, b.yield_)


def test_tracks_join_truth_by_identity_not_by_index(base_study):
    """Fails if a track is credited to the planet at its own index."""
    v = verdict(*hand_scored_log(base_study, mis_associated=1, duplicate=1, missed=1))
    assert [float(v.counts[k][0]) for k in (MISASSOCIATED, DUPLICATE, MISSED)] == [
        1.0,
        1.0,
        1.0,
    ]
    assert v.counts["met"][0] == 2
    assert v.yield_[0] == pytest.approx(1.4)
    assert v.wrong[0] == 2

    result, study = hand_scored_log(base_study)
    payload = result.carry.log.payload
    catalog = unpack(catalog_template(N_SLOTS, 2), payload[0, 0])
    catalog = dataclasses.replace(
        catalog,
        params=dataclasses.replace(
            catalog.params,
            ln_period_d=catalog.params.ln_period_d.at[2].set(jnp.log(1.0)),
        ),
    )
    encoded = pack(catalog)
    log = dataclasses.replace(
        result.carry.log, payload=payload.at[0, 0, : encoded.size].set(encoded)
    )
    empty = jnp.broadcast_to(
        pack(CRITERION.init_state(N_SLOTS, 2)), result.carry.belief.state.shape
    )
    belief = dataclasses.replace(result.carry.belief, state=empty)
    result.carry = dataclasses.replace(result.carry, log=log, belief=belief)
    unsupported = verdict(result, study)
    assert unsupported.reason[0, 0, 0] == "unsupported"
    assert unsupported.reason[0, 0, 1] == ""
    assert unsupported.yield_[0] == pytest.approx(1.4)


def test_the_verdict_reads_the_observer_summary_not_the_policy_summary(scored_case):
    """Fails if the mission label is read from the policy's summary."""
    result, study = scored_case
    initial = jnp.broadcast_to(
        pack(CRITERION.init_state(N_SLOTS, 2)), result.carry.belief.state.shape
    )
    empty = dataclasses.replace(result.carry.belief, state=initial)
    observer_only = SimpleNamespace(
        carry=dataclasses.replace(result.carry, pbelief=empty)
    )
    policy_only = SimpleNamespace(
        carry=dataclasses.replace(
            result.carry, belief=empty, pbelief=result.carry.belief
        )
    )
    got, none = verdict(observer_only, study), verdict(policy_only, study)
    np.testing.assert_array_equal(got.mission_label[0, 0, :4], 0)
    np.testing.assert_array_equal(none.mission_label, -1)


def test_the_objective_uses_true_counts_and_catalog_claims(scored_case):
    """Fails if JIT scoring uses measured signal, track index, or posterior claims."""
    result, study = scored_case
    altered = permute_detections(result, (5, 4, 3, 2, 1, 0))
    payload = altered.carry.log.payload
    catalog = unpack(catalog_template(N_SLOTS, 2), payload[0, 0])
    catalog = dataclasses.replace(
        catalog,
        params=dataclasses.replace(
            catalog.params, radius_rearth=catalog.params.radius_rearth.at[0].set(3.0)
        ),
    )
    encoded = pack(catalog)
    payload = payload.at[0, 0, : encoded.size].set(encoded)
    for index in (1, 2):
        row = unpack(measurement_template(N_SLOTS, 2), payload[0, index])
        row = dataclasses.replace(row, counts=jnp.zeros_like(row.counts))
        encoded = pack(row)
        payload = payload.at[0, index, : encoded.size].set(encoded)
    log = dataclasses.replace(altered.carry.log, payload=payload)
    altered.carry = dataclasses.replace(altered.carry, log=log)
    got = verdict(altered, study)
    assert got.yield_[0] == pytest.approx(0.7)
    one = jax.tree.map(lambda x: x[0], altered.carry)
    score = eqx.filter_jit(CriterionObjective().evaluate)(one, study)
    assert float(score) == pytest.approx(got.yield_[0])


@pytest.fixture(scope="module")
def built():
    return build(budget_d=9.0)


@pytest.mark.slow(reason="compiles the chunked refresh driver once")
def test_the_activity_table_agrees_with_the_log_and_the_ledger(built):
    """Fails if charged days disagree with the ledger or rows use the wrong kind."""
    source, study, _, _, _, run = built
    table = br.activity_table(run, study)
    rows = row_log(run, study)
    for w in range(rows["world"].max() + 1):
        for mode, spec in enumerate(study.channel_model.mode_table.modes):
            mine = (rows["world"] == w) & (rows["mode"] == mode)
            expected = rows["occupied_s"][mine].sum() / br.SECONDS_PER_DAY
            assert table.star[f"t_{spec.kind}"][w].sum() == pytest.approx(expected)
    np.testing.assert_allclose(
        (table.star["t_imaging"] + table.star["t_spectrograph"]).sum(1),
        table.world["charged"],
        atol=1e-9,
    )
    np.testing.assert_array_equal(table.star["dropped"], source.dropped)
    codes = np.asarray(run.trace.admit_code)
    np.testing.assert_array_equal(
        table.star["no_funds"].sum(1),
        (codes == AdmissionCode.NO_FUNDS).reshape(codes.shape[0], -1).sum(1),
    )


@pytest.fixture(scope="module")
def source_case(base_study):
    source = eqx.tree_at(
        lambda source: source.dropped,
        base_study.world_source,
        jnp.asarray([[1, 2], [3, 4]], I64),
    )
    study = dataclasses.replace(base_study, world_source=source)
    truths = study.world_source.take(jnp.arange(len(study.world_source.dropped)))
    carries = []
    for w in range(len(study.world_source.dropped)):
        truth = jax.tree.map(lambda x, w=w: x[w], truths)
        carry = init_carry(study, None)
        for star in range(study.caps.n_subjects):
            encoded = study.channel_model._catalog_row(truth, star, 0, None, None)
            rec = dataclasses.replace(
                record_none(study.caps),
                valid=jnp.asarray(True),
                kind=jnp.asarray(br.KIND_CATALOG, I64),
                status=jnp.asarray(RecordStatus.COMPLETED, I64),
                subject=jnp.asarray(star, I64),
                payload=jnp.pad(encoded, (0, study.caps.payload_width - encoded.size)),
            )
            carry, _ = write_log(carry, rec, study.caps)
        carries.append(carry)
    result = SimpleNamespace(
        carry=jax.tree.map(lambda *xs: jnp.stack(xs), *carries),
        trace=jax.tree.map(
            lambda x: jnp.broadcast_to(x, (len(carries), 1, *x.shape)),
            empty_trace(study.caps),
        ),
    )
    return result, study


def test_activity_refuses_a_subset_or_permuted_source_batch(source_case):
    """Fails if dropped counts are silently attached to another world's catalog."""
    result, study = source_case
    table = br.activity_table(result, study)
    np.testing.assert_array_equal(table.star["dropped"], study.world_source.dropped)
    for order in ([1, 0], [0]):
        altered = SimpleNamespace(
            carry=jax.tree.map(
                lambda x, order=order: x[jnp.asarray(order)], result.carry
            ),
            trace=jax.tree.map(
                lambda x, order=order: x[jnp.asarray(order)], result.trace
            ),
        )
        with pytest.raises(ValueError, match="source order"):
            br.activity_table(altered, study)


class RequestStar(AbstractPolicy):
    """Request an imaging observation of the declared star."""

    subject: jax.Array

    def decide(self, pstate, view, study, key):
        return (
            effects(study.caps, [EffectKind.OBSERVE], subject=self.subject),
            pstate,
            jnp.asarray(0.0),
        )


def test_no_funds_belongs_to_the_requested_star_not_the_waking_star(source_case):
    """Fails if a wake on star 0 charges a refused request for star 1 to star 0."""
    result, study = source_case
    carry = jax.tree.map(lambda x: x[0], result.carry)
    carry = dataclasses.replace(
        carry,
        ledger=dataclasses.replace(
            carry.ledger, balance=jnp.zeros_like(carry.ledger.balance)
        ),
    )
    for requested in (1, 0):
        policy = RequestStar(jnp.asarray(requested, I64))
        view = snapshot_view(carry, Audience.POLICY, study.caps)
        eff, _, u = policy.decide(
            carry.pstate, view, study.decision_view(), jax.random.PRNGKey(0)
        )
        refused, codes = admit(carry, eff, study)
        assert codes[0] == AdmissionCode.NO_FUNDS
        trace = finish_trace(
            empty_trace(study.caps), eff, codes, u, True, refused, study
        )
        unwoken = finish_trace(
            empty_trace(study.caps),
            mask_effects(eff, False),
            jnp.full_like(codes, AdmissionCode.NONE),
            u,
            False,
            refused,
            study,
        )
        np.testing.assert_array_equal(unwoken.eff_subject, 0)
        trace = dataclasses.replace(trace, subject=jnp.asarray(0, I64))
        traced = SimpleNamespace(
            carry=result.carry,
            trace=jax.tree.map(lambda x: jnp.broadcast_to(x, (2, 1, *x.shape)), trace),
        )
        totals = br.activity_table(traced, study)
        expected = np.zeros((2, 2))
        expected[:, requested] = 1
        np.testing.assert_array_equal(totals.star["no_funds"], expected)
